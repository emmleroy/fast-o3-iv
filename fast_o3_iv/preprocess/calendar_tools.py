import cftime
import xarray as xr
import numpy as np


def time_set_monthstart(ds, deep=False):
    """Modified from 
    https://github.com/NCAR/iPOGS/blob/c2cca9f398546729dbf7b08fe0baf78aa9a3dcaf/
    notebooks/AMOCz_0.1deg_RCP_loop_time.ipynb

    Edits:
    - check if time index needs to be shifted 
    - keep the existing datetime class
    """
    ds = ds.copy(deep=deep)
    idx = ds.get_index('time')

    # Check if timestamps are labeled as the next month’s start
    feb_to_jan = (int(idx[0].month) == 2) and (int(idx[-1].month) == 1)
    # Check if timestamps are anchored at day 15 (mid-month) or day 31 (end-month)
    mid_or_end_month = (int(idx[0].day) == 15) or (int(idx[0].day) == 31) # assumes jan

    if (feb_to_jan or mid_or_end_month): # check if timestamps need shifting
        year = ds['time'].dt.year
        month = ds['time'].dt.month
        new_year = xr.where(month == 1, year - 1, year)
        new_month = xr.where(month == 1, 12, month - 1)

        # Preserve the existing datetime class (cftime or python datetime)
        cls = type(ds.get_index('time')[0])
        nmonths = len(new_month)
        newtime = [cls(int(new_year[i]), int(new_month[i]), 1) for i in range(nmonths)]
        
        return ds.assign_coords({'time': newtime})
    return ds

def get_calendar(time_array):
        """extract CF time encoding from a raw dataset's time variable"""
        calendar = time_array.encoding.get("calendar") or time_array.attrs.get("calendar")
        return calendar
        
def interp_each_month_linearly(ds):
    """For future CMIP6 emissions that are prescribed decadally,
    linearly interpolate between anchor years for each month separately,
    preserving the seasonal cycle.

    Note: CESM2 uses noleap, so output timestamps use cftime.DatetimeNoLeap.
    """

    calendar = get_calendar(ds['time'])

    if calendar not in ('noleap'):
        raise ValueError("Only noleap calendars supported for now.")

    tvals = ds["time"].values
    years = np.array([t.year for t in tvals])
    months = np.array([t.month for t in tvals])

    def interp_one_month(month):
        month_mask = months == month
        sel = da.isel(time=month_mask)

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