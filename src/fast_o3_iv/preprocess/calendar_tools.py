import cftime
import xarray as xr
import numpy as np


def time_set_monthstart(ds, deep=False):
    ds = ds.copy(deep=deep)
    idx = ds.get_index("time")

    feb_to_jan = (idx[0].month == 2) and (idx[-1].month == 1)
    mid_or_end = idx[0].day in (15, 16, 30, 31)

    if not (feb_to_jan or mid_or_end):
        return ds

    year = ds["time"].dt.year
    month = ds["time"].dt.month

    if feb_to_jan:
        new_year = xr.where(month == 1, year - 1, year)
        new_month = xr.where(month == 1, 12, month - 1)
    else:
        new_year = year
        new_month = month

    cls = type(idx[0])
    newtime = [cls(int(y), int(m), 1) for y, m in zip(new_year.values, new_month.values)]
    return ds.assign_coords(time=("time", newtime))

def get_time_encoding(time_array):
        """extract CF time encoding from a raw dataset's time variable"""
        units = time_array.encoding.get("units") or time_array.attrs.get("units")
        calendar = time_array.encoding.get("calendar") or time_array.attrs.get("calendar")
        return units, calendar
        
def interp_each_month_linearly(ds):
    """For future CMIP6 emissions that are prescribed decadally,
    linearly interpolate between anchor years for each month separately,
    preserving the seasonal cycle.

    Note: CESM2 uses noleap, so output timestamps use cftime.DatetimeNoLeap.
    """

    _, calendar = get_time_encoding(ds['time'])

    if calendar not in ('noleap', '365_day'): # both of these are the same --> each has 365 days, no leap years
        raise ValueError("Only noleap/365_day calendars supported for now.")

    tvals = ds["time"].values
    years = np.array([t.year for t in tvals])
    months = np.array([t.month for t in tvals])

    def interp_one_month(month):
        month_mask = months == month
        sel = ds.isel(time=month_mask)

        sel_years = years[month_mask]
        sel = sel.rename({"time": "year"}).assign_coords(year=("year", sel_years))

        start_year = int(sel_years.min())
        end_year = int(sel_years.max())
        target_years = np.arange(start_year, end_year + 1)

        sel_interp = sel.interp(year=target_years)

        new_time = [cftime.DatetimeNoLeap(tyear, month, 1) for tyear in target_years]
        return sel_interp.rename({"year": "time"}).assign_coords(time=("time", new_time))
 
    interp_all_months = [interp_one_month(m) for m in range(1, 13)]
    return xr.concat(interp_all_months, dim="time").sortby("time")

def to_localtime(ds, lon_name="lon", lat_name="lat"):
    """Convert hourly dataset at UTC to Local Standard Time (no DST) at every grid cell.
-    Note: works with datetime64 and cftime calendars since shift is index-based.
-    Note: originally adapted from https://github.com/barronh/pycmaq/
-    LST is defined by longitude.
-    """

    ds0 = ds.copy(deep=False) # keep original ds unchanged

    # 1. convert [0,360] to [-180, 180]
    # this is so that timeznne offsets are signed around Greenwich (0°)
    # west longitudes should have negative hours
    # east longitudes hsould have positive hours
    lon = ds0[lon_name]
    lon180 = xr.where(lon > 180, lon - 360, lon)

    # 2. round longitude to the nearest multiple of 15°
    # (time zones are ~roughly assigned every 15° of longitude, 
    # since Earth rotates 360° / 24 hours = 15° per hour
    # (x + 7.5) / 15.0 --> bin longitudes into 15° timezones with nearest rounding
    # we add +7.5 so that boundaries are centered between bins (nearest-bin rounding for 15° bins)
    # np.floor( ..., -12, 14) --> bound offsets to plausible UTC range
    tz_off_1d = xr.apply_ufunc(
        lambda x: np.clip(np.floor((x + 7.5) / 15.0), -12, 14).astype(np.int8),
        lon180,
        dask="parallelized",
        output_dtypes=[np.int8],
    )

    # broadcast to (lat, lon) since tz_off_1d depends only on longitude
    _, tz_off = xr.broadcast(ds0[lat_name], tz_off_1d)

     # add tz offset as a new coordinate to the dataset
    ds1 = ds0.assign_coords(tz_off=tz_off)

    # function to shift per tz offset; positive offset means local time is ahead of UTC
    def _shift(group):
        """for a given tz offset value, grab the tz offset and shift time dimension by the offset"""
        offset = int(group["tz_off"].values.flat[0])
        if offset == 0:
            return group
        return group.shift(time=offset)

    # apply the shift time-zone offset values
    out = ds1.groupby("tz_off").map(_shift)
    return out.drop_vars("tz_off", errors="ignore")