"""Read-only native-grid ERA5 extraction, spherical derivatives and original-area support."""
from __future__ import annotations
from dataclasses import dataclass
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
import json
import numpy as np
import netCDF4
EARTH_RADIUS_M = 6371008.8
SECONDS_PER_DAY = 86400.0

def utc_datetime(value):
    """Require an exact UTC time; naive ISO/datetimes are interpreted as documented UTC."""
    if isinstance(value, np.datetime64):
        value = np.datetime_as_string(value, unit='us')
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if not isinstance(value, datetime):
        raise TypeError('utc must be datetime, np.datetime64, or ISO8601 string')
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value

def longitude_difference(lon, reference):
    """Signed physical longitude difference in [-180, 180), including dateline crossing."""
    return (np.asarray(lon, dtype=float) - float(reference) + 180.0) % 360.0 - 180.0

def great_circle_distance_km(latitude, longitude, center_lat, center_lon):
    lat = np.deg2rad(np.asarray(latitude, dtype=float))
    dl = np.deg2rad(longitude_difference(longitude, center_lon))
    dlat = lat - np.deg2rad(center_lat)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(np.deg2rad(center_lat)) * np.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M / 1000 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def native_cell_area_m2(latitude, longitude):
    """Exact spherical lat/lon cell areas at regular native grid-cell centres."""
    lat = np.asarray(latitude, dtype=float)
    lon = np.asarray(longitude, dtype=float)
    if lat.ndim != 1 or lon.ndim != 1 or len(lat) < 2 or (len(lon) < 2):
        raise ValueError('at least two one-dimensional native latitude/longitude centres required')
    dlat = abs(float(np.median(np.diff(lat))))
    dlon = abs(float(np.median(np.diff(lon))))
    if not (np.allclose(abs(np.diff(lat)), dlat, rtol=0, atol=1e-10) and np.allclose(abs(np.diff(lon)), dlon, rtol=0, atol=1e-10)):
        raise ValueError('native grid must be regular; longitude must be continuous across dateline')
    north = np.deg2rad(np.minimum(90, lat + dlat / 2))
    south = np.deg2rad(np.maximum(-90, lat - dlat / 2))
    row = EARTH_RADIUS_M ** 2 * np.deg2rad(dlon) * (np.sin(north) - np.sin(south))
    return np.broadcast_to(row[:, None], (len(lat), len(lon))).copy()

def _as_float(a):
    return np.asarray(np.ma.filled(np.ma.asarray(a, dtype=float), np.nan), dtype=float)

@dataclass
class NativeBlock:
    key: str
    variable: str
    utc: str
    units: str
    pressure_level_pa: float | None
    latitude: np.ndarray
    longitude: np.ndarray
    latitude_indices: np.ndarray
    longitude_indices: np.ndarray
    raw: np.ndarray
    values: np.ndarray
    support_mask: np.ndarray
    sp_pa: np.ndarray | None
    cell_area_m2: np.ndarray
    distance_km: np.ndarray
    core_mask: np.ndarray
    center_lat: float
    center_lon: float
    radius_km: float
    halo: int
    metadata: dict

    def annulus(self, inner_km=0.0, outer_km=None):
        outer_km = self.radius_km if outer_km is None else float(outer_km)
        if inner_km < 0 or outer_km > self.radius_km or outer_km < inner_km:
            raise ValueError('annulus must be within the extracted core radius')
        return (self.distance_km >= inner_km) & (self.distance_km <= outer_km)

class NativeFieldReader:
    """Small LRU of read-only NetCDF handles; one caller/IO worker; context-manager friendly."""

    def __init__(self, variable_map, max_open=6, max_block_cells=250000, chunk_cache_bytes=8 * 1024 ** 2):
        if isinstance(variable_map, (str, Path)):
            variable_map = json.loads(Path(variable_map).read_text(encoding='utf-8-sig'))
        self.map = variable_map.get('paths', variable_map)
        self.max_open = int(max_open)
        self.max_block_cells = int(max_block_cells)
        if self.max_open < 2:
            raise ValueError('max_open must be at least two for field and SP')
        self.chunk_cache_bytes = int(chunk_cache_bytes)
        self._files = OrderedDict()
        self._coords = {}
        self._times = {}
        self.read_log = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        for ds in self._files.values():
            ds.close()
        self._files.clear()

    def _open(self, key):
        if key not in self.map:
            raise KeyError(f'unavailable audited field: {key}')
        p = self.map[key]['path']
        if p in self._files:
            self._files.move_to_end(p)
            return self._files[p]
        while len(self._files) >= self.max_open:
            _, ds = self._files.popitem(last=False)
            ds.close()
        ds = netCDF4.Dataset(p, 'r')
        vn = self.map[key]['variable']
        if vn not in ds.variables:
            ds.close()
            raise ValueError(f'{p}: expected {vn}')
        v = ds[vn]
        v.set_var_chunk_cache(size=self.chunk_cache_bytes, nelems=1009, preemption=0.5)
        self._files[p] = ds
        return ds

    def _time_index(self, key, utc):
        cfg = self.map[key]
        ds = self._open(key)
        tn = cfg['time_variable']
        if tn is None:
            return None
        tv = ds[tn]
        if key == 'lsm' and len(tv) == 1:
            return 0
        dt = utc_datetime(utc)
        if dt.minute or dt.second or dt.microsecond:
            raise ValueError(f'not an exact source UTC: {dt.isoformat()}')
        target = float(netCDF4.date2num(dt, tv.units, getattr(tv, 'calendar', 'standard')))
        path = cfg['path']
        if path not in self._times:
            self._times[path] = np.asarray(tv[:], dtype=float)
        t = self._times[path]
        ix = int(np.searchsorted(t, target))
        if ix >= len(t) or t[ix] != target:
            raise KeyError(f'{key}: exact UTC {dt.isoformat()} absent; no nearest-time fallback')
        return ix

    def _coordinates(self, key):
        p = self.map[key]['path']
        if p not in self._coords:
            ds = self._open(key)
            self._coords[p] = (np.asarray(ds[self.map[key].get('latitude_variable', 'latitude')][:], dtype=float), np.asarray(ds[self.map[key].get('longitude_variable', 'longitude')][:], dtype=float))
        return self._coords[p]

    def _grid_check(self, key, reference):
        lat, lon = self._coordinates(key)
        if not (np.array_equal(lat[reference.latitude_indices], reference.latitude) and np.allclose(longitude_difference(lon[reference.longitude_indices], reference.center_lon), longitude_difference(reference.longitude, reference.center_lon), rtol=0, atol=1e-10)):
            raise ValueError(f'{key}: physical coordinate mismatch; regridding is not automatic')

    def _read(self, key, utc, ilat, ilon):
        if len(ilat) * len(ilon) > self.max_block_cells:
            raise ValueError('spatial request exceeds bounded cell limit')
        ti = self._time_index(key, utc)
        ds = self._open(key)
        cfg = self.map[key]
        v = ds[cfg['variable']]
        tn = cfg['time_variable']
        idx = []
        for name in v.dimensions:
            if name == tn:
                idx.append(ti)
            elif name == cfg.get('latitude_variable', 'latitude'):
                idx.append(np.asarray(ilat, dtype=int))
            elif name == cfg.get('longitude_variable', 'longitude'):
                idx.append(np.asarray(ilon, dtype=int))
            elif len(ds.dimensions[name]) == 1:
                idx.append(0)
            else:
                raise ValueError(f'unhandled non-singleton dimension {name}: {key}')
        a = _as_float(v[tuple(idx)])
        if a.shape != (len(ilat), len(ilon)):
            raise ValueError(f'unexpected spatial dimensions {a.shape}')
        self.read_log.append(dict(key=key, utc=utc_datetime(utc).isoformat() + 'Z', time_index=ti, shape=list(a.shape), n_requested=int(a.size)))
        return a

    def read_box(self, key, utc, center_lat, center_lon, radius_km, halo=1, apply_sp=True):
        center_lat = float(center_lat)
        center_lon = float(center_lon)
        radius_km = float(radius_km)
        halo = int(halo)
        if not -90 < center_lat < 90 or not 0 < radius_km < 10000 or halo < 0:
            raise ValueError('invalid centre, radius or halo')
        lat, lon = self._coordinates(key)
        delta = radius_km * 1000 / EARTH_RADIUS_M
        deg = np.rad2deg(delta)
        il = np.flatnonzero((lat >= max(-90, center_lat - deg)) & (lat <= min(90, center_lat + deg)))
        if not len(il):
            il = np.array([int(np.argmin(abs(lat - center_lat)))])
        il = np.arange(max(0, int(il.min()) - halo), min(len(lat), int(il.max()) + halo + 1))
        pole_crossing = abs(center_lat) + deg >= 90
        lon_half = 180.0 if pole_crossing else np.rad2deg(np.arcsin(np.clip(np.sin(delta) / np.cos(np.deg2rad(center_lat)), 0, 1)))
        rel = longitude_difference(lon, center_lon)
        order = np.argsort(rel)
        ordered = rel[order]
        hit = np.flatnonzero(abs(ordered) <= lon_half)
        if not len(hit):
            hit = np.array([int(np.argmin(abs(ordered)))])
        left = max(0, int(hit.min()) - halo)
        right = min(len(lon), int(hit.max()) + halo + 1)
        jl = order[left:right]
        physlon = center_lon + ordered[left:right]
        if len(il) < 3 or len(jl) < 3:
            raise ValueError('need at least3 native cells per axis including halo')
        la = lat[il]
        distance = great_circle_distance_km(la[:, None], physlon[None, :], center_lat, center_lon)
        cfg = self.map[key]
        level = cfg.get('pressure_level_hpa')
        p_pa = None if level is None else float(level) * 100
        raw = self._read(key, utc, il, jl)
        sp = None
        mask = np.isfinite(raw)
        if apply_sp and p_pa is not None:
            splat, splon = self._coordinates('sp')
            if not (np.array_equal(splat[il], la) and np.allclose(longitude_difference(splon[jl], center_lon), longitude_difference(physlon, center_lon), rtol=0, atol=1e-10)):
                raise ValueError('SP and field coordinates differ')
            sp = self._read('sp', utc, il, jl)
            mask &= np.isfinite(sp) & (sp >= p_pa)
        values = np.where(mask, raw, np.nan)
        return NativeBlock(key, cfg['variable'], utc_datetime(utc).isoformat() + 'Z', cfg['units'], p_pa, la, physlon, il, jl, raw, values, mask, sp, native_cell_area_m2(la, physlon), distance, distance <= radius_km, center_lat, center_lon, radius_km, halo, dict(path=cfg['path'], applied_sp_mask=bool(apply_sp and p_pa is not None), raw_units=cfg['units'], denominator='original geometric cells before SP/finite-value filtering', time_index=self.read_log[-1]['time_index']))

    def read_aligned(self, key, utc, reference, apply_sp=True):
        self._grid_check(key, reference)
        cfg = self.map[key]
        raw = self._read(key, utc, reference.latitude_indices, reference.longitude_indices)
        level = cfg.get('pressure_level_hpa')
        p_pa = None if level is None else float(level) * 100
        sp = None
        mask = np.isfinite(raw)
        if apply_sp and p_pa is not None:
            self._grid_check('sp', reference)
            if reference.sp_pa is not None and utc_datetime(reference.utc) == utc_datetime(utc):
                sp = reference.sp_pa
            else:
                sp = self._read('sp', utc, reference.latitude_indices, reference.longitude_indices)
            mask &= np.isfinite(sp) & (sp >= p_pa)
        return NativeBlock(key, cfg['variable'], utc_datetime(utc).isoformat() + 'Z', cfg['units'], p_pa, reference.latitude.copy(), reference.longitude.copy(), reference.latitude_indices.copy(), reference.longitude_indices.copy(), raw, np.where(mask, raw, np.nan), mask, sp, reference.cell_area_m2.copy(), reference.distance_km.copy(), reference.core_mask.copy(), reference.center_lat, reference.center_lon, reference.radius_km, reference.halo, dict(path=cfg['path'], applied_sp_mask=bool(apply_sp and p_pa is not None), denominator='original geometric cells'))

def _field_grid(field, latitude, longitude):
    a = _as_float(field)
    lat = np.asarray(latitude, dtype=float)
    lon = np.asarray(longitude, dtype=float)
    if a.shape != (len(lat), len(lon)) or min(a.shape) < 3:
        raise ValueError('field must match native2D grid with halo')
    for x in (lat, lon):
        d = np.diff(x)
        if not (np.all(d > 0) or np.all(d < 0)):
            raise ValueError('coordinates must be strictly monotonic and longitude unwrapped')
    return (a, np.deg2rad(lat), np.deg2rad(lon))

def spherical_gradient(field, latitude, longitude):
    """(east,north) centred gradients per metre; requires both immediate neighbours and centre.
    Outer one-cell boundary is NaN: no fabricated one-sided derivative outside the halo.
    """
    a, phi, lam = _field_grid(field, latitude, longitude)
    east = np.gradient(a, lam, axis=1, edge_order=2) / (EARTH_RADIUS_M * np.cos(phi)[:, None])
    north = np.gradient(a, phi, axis=0, edge_order=2) / EARTH_RADIUS_M
    east[:, [0, -1]] = np.nan
    north[[0, -1], :] = np.nan
    east[~np.isfinite(a)] = np.nan
    north[~np.isfinite(a)] = np.nan
    pole = abs(np.cos(phi)) < 1e-08
    east[pole, :] = np.nan
    north[pole, :] = np.nan
    return (east, north)

def spherical_divergence(u, v, latitude, longitude):
    """Horizontal divergence s-1 using native metric terms ∂u/∂λ+∂(v cosφ)/∂φ."""
    u, phi, lam = _field_grid(u, latitude, longitude)
    v, _, _ = _field_grid(v, latitude, longitude)
    a = np.gradient(u, lam, axis=1, edge_order=2)
    b = np.gradient(v * np.cos(phi)[:, None], phi, axis=0, edge_order=2)
    result = (a + b) / (EARTH_RADIUS_M * np.cos(phi)[:, None])
    result[[0, -1], :] = np.nan
    result[:, [0, -1]] = np.nan
    result[~(np.isfinite(u) & np.isfinite(v))] = np.nan
    result[abs(np.cos(phi)) < 1e-08, :] = np.nan
    return result

def storm_relative_advection(q, u, v, c_east, c_north, latitude, longitude):
    """−[(u−c_east)dq/dx+(v−c_north)dq/dy], kg kg-1 s-1.
    c must use pre/current motion defined by caller; missing motion remains missing.
    """
    qe, qn = spherical_gradient(q, latitude, longitude)
    return -((_as_float(u) - np.asarray(c_east)) * qe + (_as_float(v) - np.asarray(c_north)) * qn)

def vertical_advection_700(q600, q700, q850, omega700):
    """−omega dq/dp at700hPa. Nonuniform three-point derivative using600/700/850hPa.
    Pressures in Pa; omega Pa s-1; all3q values must be supported.
    """
    qs = np.stack([_as_float(q600), _as_float(q700), _as_float(q850)])
    dp = np.gradient(qs, np.array([60000.0, 70000.0, 85000.0]), axis=0, edge_order=2)[1]
    out = -_as_float(omega700) * dp
    return np.where(np.isfinite(qs).all(axis=0) & np.isfinite(omega700), out, np.nan)

def moving_frame_tendency(q_before, q_after, delta_seconds):
    """Centred trajectory-following difference from native-derived/consistently collocated fields.
    Caller must provide the two actual samples in the same moving-frame locations;
    this function does not interpolate positions or fabricate absent track support.
    """
    if delta_seconds <= 0:
        raise ValueError('positive actual elapsedseconds required')
    return (_as_float(q_after) - _as_float(q_before)) / float(delta_seconds)

def shear_components(u_upper, v_upper, u_lower, v_lower):
    """Signed upper-minus-lower components and magnitude, all m s-1."""
    du = _as_float(u_upper) - _as_float(u_lower)
    dv = _as_float(v_upper) - _as_float(v_lower)
    return (du, dv, np.hypot(du, dv))

def rotate_vectors(east, north, angle_rad):
    """Components in a fixed frame whose +x axis is angle_rad CCW from geographic east."""
    e = np.asarray(east, dtype=float)
    n = np.asarray(north, dtype=float)
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return (c * e + s * n, -s * e + c * n)

def to_fixed_frame(x_east, y_north, u_east, v_north, angle_rad):
    """Rotate physical coordinates and vectors with the SAME fixed event angle."""
    x, y = rotate_vectors(x_east, y_north, angle_rad)
    u, v = rotate_vectors(u_east, v_north, angle_rad)
    return (x, y, u, v)

def hemisphere_reflection(x, y, u, v, hemisphere):
    """Optional explicit equatorial reflection. Geographic x/east unchanged; y/north mirrored.
    It is a geometric operation, NOT evidence that NH/SH structures may be pooled.
    Under this transform vorticity changes sign. NH=+1,SH=−1.
    """
    if hemisphere not in (-1, 1):
        raise ValueError('hemisphere must be−1 or+1')
    return (np.asarray(x), hemisphere * np.asarray(y), np.asarray(u), hemisphere * np.asarray(v))

def area_weighted_support(values, cell_area_m2, domain_mask, min_fraction=0.9):
    """Valid-area weighted mean gated by ORIGINAL geometric area, never renormalised QC denominator."""
    a = _as_float(values)
    w = np.asarray(cell_area_m2, dtype=float)
    domain = np.asarray(domain_mask, dtype=bool)
    if not a.shape == w.shape == domain.shape:
        raise ValueError('values/area/domain shapes differ')
    if not 0 <= min_fraction <= 1:
        raise ValueError('min_fraction must lie[0,1]')
    if np.any(domain & (~np.isfinite(w) | (w <= 0))):
        raise ValueError('domain original-area cells must be finite positive')
    denom = float(w[domain].sum())
    valid = domain & np.isfinite(a)
    num = float(w[valid].sum())
    fraction = num / denom if denom > 0 else np.nan
    passed = denom > 0 and fraction >= min_fraction
    mean = float(np.sum(a[valid] * w[valid]) / num) if num > 0 else np.nan
    return dict(mean=mean if passed else np.nan, raw_supported_mean=mean, valid_area_m2=num, original_area_m2=denom, valid_fraction=fraction, valid_cells=int(valid.sum()), original_cells=int(domain.sum()), support_pass=bool(passed))
