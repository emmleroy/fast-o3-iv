import cftime
import xarray as xr

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