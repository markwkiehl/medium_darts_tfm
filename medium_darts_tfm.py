#
#   Written by:  Mark W Kiehl
#   http://mechatronicsolutionsllc.com/
#   http://www.savvysolutions.info/savvycodesolutions/
#
#   Version 0.0.2
#
#   0.0.1   revised get_darts_model_splits()
#   0.0.2   Added two examples that demonstrate how to use get_darts_tfm_arguments_optimized_via_backtesting()




def sec_to_dhms(s):
    #print(sec_to_dhms(3600*25+62.2))
    
    d = s // (24 * 3600) 

    s = s % (24 * 3600) 
    h = s // 3600

    s %= 3600
    m = s // 60

    s %= 60

    long_str = str(d) + " days " + str(h) + " hours " + str(m) + " min " + str(round(s,1)) + " sec"
    
    short_str = ""
    if d > 0: short_str += str(d) + " days "
    if h > 0: short_str += str(h) + " hours "
    if m > 0: short_str += str(m) + " min "
    short_str += str(round(s,1)) + " sec"

    return short_str



def power_by_base(num, base):
  # Returns the power for the value num and the base
  # Ex. 
  #print(power_by_base(12,2))      # 8     2^3
  #print(power_by_base(24,2))      # 16    2^4
  #print(power_by_base(33.3,2))    # 32    2^5
  #print(power_by_base(150,2))     # 128   2^7
  #print(power_by_base(365,2))     # 256   2^8
  i = 1
  while i < num: i *= base
  i //= base
  return i


def next_power_by_base(num, base):
  # Returns the next power for the value num and the base
  # Ex. 
  # print(next_power_by_base(31.1,2))       # 32
  # print(next_power_by_base(33.3,2))       # 64
  i = 1
  while i < num: i *= base
  return i    


def powers_of_two_in_range(start=None, end=None):
    # Returns a ascending sorted list of powers of two with values between start and end (inclusive).
    # The value for end must be greater than start or an exception will be raised. 
    # Used as range of inputs for ML/AI models. 
    
    if start is None or end is None: raise Exception("ERROR: arguments start, end length for powers_of_two_in_range() cannot be None")
    if end < start: raise Exception("ERROR: end < start passed to powers_of_two_in_range()")

    i = 0
    lst = []
    done = False
    while not done:
        num = power_by_base(i,2)
        if num >= power_by_base(start,2) and num <= power_by_base(end,2) and num not in lst: #  
            lst.append(num)
        #print(power_by_base(start,2), power_by_base(end,2), "2^"+str(i)+"="+str(num), len(lst))
        #print(lst)
        i += 1
        if power_by_base(start,2) in lst and power_by_base(end,2) in lst: done = True
    return lst


def divisors_of_int(n=None, smallest=True):
    # Returns the divisors of integer n as a list, sorted by the smallest first.
    # If smallest=False, returns the largest divisor of n (excluding n) first, sorted decending.

    if not isinstance(n, int): raise Exception("Value Error: n is not an integer. ", n)
    if not isinstance(smallest, bool): raise Exception("Value Error: smallest must be a boolean with value of True or False")

    divisors = []
    # 1 divides everything, so start testing from 2
    for i in range(2, n):
        # If i divides n without a remainder, it is the smallest divisor
        if n % i == 0:
            divisors.append(i)
    
    if len(divisors) == 0: return [n]

    if smallest == True:
        return divisors
    else:
        # get largest divisor
        divisors.sort(reverse=True)
        return divisors



def sine_gaussian_noise_covariate(length=160, add_trend=False, include_covariate=False, multivariate=False, start=None):
    # Generates Pandas dataframe of a noisy sine wave with optional trend and optional additional covariate column.
    # The noisy sine wave will have a seasonal period = 10.  
    # The noisy signal will have an amplitude generally greater than +/- 1.0
    # All data of numpy float32 for model effiency. 
    # The index is a datetime index that defaults to starting at 2000-01-01.  Assign alternative value as start=pd.Timestamp(year=2001, month=12, day=27)
    # If multivariate==True, then 3x different signals are returned, and one will be the noise sine wave.
    # If add_trend==True, then a trend will be applied to the univariate/multivariate signals, but not the covariate.  

    # Series created by taking the some of a sine wave and a gaussian noise series.
    # The intensity of the gaussian noise is also modulated by a sine wave (with 
    # a different frequency). 
    # The effect of the noise gets stronger and weaker in an oscillating fashion.
    # Derived from:  https://unit8co.github.io/darts/examples/08-DeepAR-examples.html#Variable-noise-series
    
    from darts import TimeSeries
    import darts.utils.timeseries_generation as tg
    from darts.utils.missing_values import fill_missing_values
    import pandas as pd
    import numpy as np

    if start is None: start = pd.Timestamp("2000-01-01")
    if not isinstance(start, pd.Timestamp): raise Exception("Value Error: start is not a valid timestamp.")

    trend = tg.linear_timeseries(length=length, end_value=4, start=start)    # end_value=4
    season1 = tg.sine_timeseries(length=length, value_frequency=0.05, value_amplitude=1.0, start=start)  # value_amplitude=1.0
    noise = tg.gaussian_timeseries(length=length, std=0.6, start=start)  
    noise_modulator = (
        tg.sine_timeseries(length=length, value_frequency=0.02, start=start)
        + tg.constant_timeseries(length=length, value=1, start=start)
    ) / 2
    covariates = noise_modulator.astype(np.float32)      # convert to numpy 32 bit float for model efficiency
    noise = noise * noise_modulator
    if add_trend:
        ts = sum([noise, season1, trend])
    else:
        ts = sum([noise, season1])
    ts = ts.astype(np.float32)      # convert to numpy 32 bit float for model efficiency

    if multivariate == True:
        ts1 = ts.copy()
        del ts
        
        # Create a second TimeSeries from df that is smooth and with different scale and a mean offset
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=4)
        df = ts1.pd_dataframe()
        ds = df.rolling(window=indexer, min_periods=4).sum()
        ts2 = TimeSeries.from_series(pd_series=ds) 
        ts2 = ts2.rescale_with_value(0.5)
        ts2 = ts2 + 1.0
        ts2 = fill_missing_values(ts2)
        del indexer, ds, df

        # Create a third TimeSeries from ts1 using the .map() TimeSeries method
        ts3 = ts1.map(lambda x: x ** 2) 
        ts3 = ts3 - 1.1        # apply a mean offset

        # Create a multivariate TimeSeries from the three univariate time series.
        ts = ts1.stack(ts2)
        ts = ts.stack(ts3)
        del ts1, ts2, ts3

        # rename the columns
        ts = ts.with_columns_renamed(ts.columns, ['ts1','ts2','ts3'])


    df = ts.pd_dataframe()

    if include_covariate:
        ds = covariates.pd_series()
        df = df.assign(new_col_ds=ds)
        if ts.width == 1:
            df.rename(columns={df.columns[0]: 'noisy_sin', df.columns[1]: 'covariate'}, inplace=True, errors="raise")
        else:
            df.rename(columns={df.columns[3]: 'covariate'}, inplace=True, errors="raise")
    else:
        if ts.width == 1: df.rename(columns={df.columns[0]: 'noisy_sin'}, inplace=True, errors="raise")
    df.index.rename("Date", inplace=True)

    return df



def ts_seasonal_periods(ts=None, sort_ascending=True, unique=False):
    # Returns list of the seasonal periods of the Darts TimeSeries ts.
    # Most useful if 'ts' is a multivariate series. 
    # Returns empty list if no seasonality found. 

    # Get the seasonal periods of all three components/columns in the multivariate series 'ts':
    # print("seasonal_periods:", ts_seasonal_periods(ts=ts, sort_ascending=False, unique=False))    # seasonal_periods: [20, 20, 9]
    
    # Get the minimum seasonal period in the multivariate series 'ts' with three components/columns:
    # print("seasonal_periods:", ts_seasonal_periods(ts=ts, sort_ascending=True, unique=True))        # seasonal_periods: [9, 20]

    from darts import TimeSeries
    from darts.utils.statistics import check_seasonality

    if not isinstance(ts, TimeSeries): raise Exception("ts is not a Darts TimeSeries")

    seasonal_periods = []

    """
    if ts.width == 1:
        is_seasonal, seasonal_period = check_seasonality(ts=ts)
        if is_seasonal: 
            seasonal_periods.append(seasonal_period)
    """
        
    # ts.width > 1
    for idx in range(0, ts.width):
        series = ts.univariate_component(idx)
        is_seasonal, seasonal_period = check_seasonality(ts=series)
        if is_seasonal: 
            if unique == True:
                if not seasonal_period in seasonal_periods: seasonal_periods.append(seasonal_period)
            else:
                seasonal_periods.append(seasonal_period)

    if sort_ascending: seasonal_periods.sort()
    # Convert from numpy.int64 to Python int
    seasonal_periods = [int(x) for x in seasonal_periods]
    return seasonal_periods



def get_idx_datetime_attr(freq_str=None):
    # Maps the Darts / Pandas datetime index 
    #   Darts: .freq string '<Day>' and .freq_str 'D' strings 
    #   Pandas: .freq and .freqstr strings
    # to the Darts darts.utils.timeseries_generation.datetime_attribute_timeseries 'attribute' argument corresponds
    # to the Pandas DatetimeIndex string attributes, e.g. “month”, “weekday”, “day”, “hour”, “minute”, “second”.
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DatetimeIndex.html#pandas.DatetimeIndex
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-period-aliases
    # Aliases A, H, T, S, L, U, and N are deprecated in favour of the aliases Y, h, min, s, ms, us, and ns
    
    # index .freq       index .freq_str         'attribute' argument for datetime_attribute_timeseries
    #                                           'year'
    #                       'MS                 'month'
    #   <Day>               'D'                 'day'
    #                       'H'                 'hour'
    #                       'T'                 'minute'
    #                       'S'                 'second'
    #                       'U'                 'microsecond'
    #                       'L'                 'nanosecond'

    if freq_str is None: raise Exception("freq_str is None!")

    freqstr_attr = {
        'D': 'day',
        'H': 'hour',
        'bh': 'hour',
        'cbh': 'hour',
        'T': 'minute',
        'min': 'minute',
        'S': 'second',
        'U': 'microsecond',
        'L': 'nanosecond',
        'MS': 'month',    
        'ME': 'month',
        'BME': 'month',
        'CBME': 'month',
        'BMS': 'month',
        'QE': 'quarter',
        'BQE': 'quarter',
        'QS': 'quarter',
        'BQS': 'quarter',
        'YE': 'year',
        'BYE': 'year',
        'YS': 'year',
        'BYS': 'year',
    }
    
    if freq_str[0:3] == 'QS-' and not freq_str in freqstr_attr: freqstr_attr[freq_str] = 'quarter'
    if freq_str[0:2] == 'W-' and not freq_str in freqstr_attr: freqstr_attr[freq_str] = 'week'
    if freq_str[len(freq_str)-1] == 'T' and not freq_str in freqstr_attr: freqstr_attr[freq_str] = 'minute'     # '10T' -> 'T' minute

    if not freq_str in freqstr_attr: raise Exception("ERROR: freq_str '" + freq_str + "' passed to get_idx_datetime_attr() is not recognized.")
    return freqstr_attr[freq_str]


def get_df_index_freq_val_unit(df=None, verbose=False):
    # Note that Pandas df.index.freqstr will return the freq string.

    # Returns the index frequency value and unit.
    # idx_freq_val, idx_freq_unit = get_df_index_freq_val_unit(df)

    # Determines (measures) the index freq and returns the value
    # along with the unit.  For a numeric index, the unit returned will
    # be None, otherwise for datetime the unit will be:
    # Offset Alias  Description
    #   D               day
    #   H               hour
    #   T               minute
    #   S               second
    #   L               millisecond
    #   U               microsecond
    #   N               nanosecond
    #  None a numeric index (not of type datetime)
    
    # Note that the Offset Alias is a TimeDelta resolution_string.
    #   https://pandas.pydata.org/docs/user_guide/timeseries.html#offset-aliases
    #   https://pandas.pydata.org/docs/reference/api/pandas.DatetimeIndex.freqstr.html
    #   https://pandas.pydata.org/docs/user_guide/timeseries.html#dateoffset-objects

    import pandas as pd
    if not isinstance(df, pd.DataFrame) and not isinstance(df, pd.Series): raise Exception("Argument 'df' passed to get_df_index_freq_val_unit() must be a Pandas dataframe or series")

    # Measure index spacing by creating a new dataframe
    # df_tmp with the difference of each ajoining index value. 
    df_tmp = df.index.to_series().diff().to_frame()

    # Drop the first row in the first column with the value of NaT (datetime) or NaN (numeric)
    df_tmp.dropna(inplace=True)

    df_tmp.rename(columns={df_tmp.columns[0]: 'Diff'}, inplace=True, errors="raise")

    # Calculate the mode for column 'Diff' and get the maximum value as mode_max.
    if verbose: print("Col 'diff' has a mode().median() of: ", df_tmp['Diff'].mode().median())    # 1 days 00:00:00
    if verbose: print("Col 'diff' has a median of: ", df_tmp['Diff'].median())      # 1 days 00:00:00

    mode = df_tmp['Diff'].mode().median()
    # Note that the mode is of type TimeDelta.  https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timedelta.html#pandas-timedelta
    #print("type(mode): ", type(mode))   # <class 'pandas._libs.tslibs.timedeltas.Timedelta'>

    # Determine if the index is of type datetime (<class 'pandas.core.indexes.datetimes.DatetimeIndex'>), or numeric (<class 'pandas.core.indexes.range.RangeIndex'>)
    if isinstance(df.index, pd.DatetimeIndex):
        if verbose: print("Dataframe index is of type datetime")
    else:
        if verbose: print("Dataframe index is of type numeric")
        return mode, ""


    #print(mode.components)  # Components(days=31, hours=0, minutes=0, seconds=0, milliseconds=0, microseconds=0, nanoseconds=0)
    #print(mode.components.days) # 31
    #print(mode.resolution_string)   # D
    # NOTE: The maximum resolution of mode.resolution_string is D
    if mode.resolution_string == "D": return mode.days, mode.resolution_string  # days
    elif mode.resolution_string == "H": return mode.components.hours, mode.resolution_string  # hours
    elif mode.resolution_string == "T": return mode.components.minutes, mode.resolution_string  # minutes
    elif mode.resolution_string == "S": return mode.seconds, mode.resolution_string  # seconds
    elif mode.resolution_string == "L": return mode.components.milliseconds, mode.resolution_string  # milliseconds
    elif mode.resolution_string == "U": return mode.microseconds, mode.resolution_string  # microseconds
    elif mode.resolution_string == "N": return mode.nanoseconds, mode.resolution_string  # nanoseconds
    else:
        raise Exception("ERROR: Unexpected .resolution_string of '" + mode.resolution_string + "'")


def index_has_gaps(df=None, verbose=False):
    # Returns True if gaps are found in the index of dataframe df
    # Works with index of type datetime or numeric

    import pandas as pd
    import math

    if not isinstance(df, pd.DataFrame) and not isinstance(df, pd.Series): raise Exception("Argument 'df' passed to index_has_gaps() must be a Pandas dataframe or series")

    # USAGE:
    #from savvy_time_series import index_has_gaps
    #print("index_has_gaps() ", index_has_gaps(df=df, verbose=True))


    # Determine if the index is of type datetime (<class 'pandas.core.indexes.datetimes.DatetimeIndex'>), or numeric (<class 'pandas.core.indexes.range.RangeIndex'>)
    if isinstance(df.index, pd.DatetimeIndex):
        if verbose: print("Dataframe index is of type datetime")
    else:
        if verbose: print("Dataframe index is of type numeric")

    # Identify gaps in the index of df by creating a new dataframe
    # df_tmp with the difference of each ajoining index value. 
    #import pandas as pd
    df_tmp = df.index.to_series().diff().to_frame()
    #            Time
    #Time
    #0.000000       NaN
    #0.000083  0.000083

    # Drop the first value in the first column with the value of NaT (datetime) or NaN (numeric)
    df_tmp.dropna(inplace=True)
    #if verbose: print(df_tmp)
    #            Time
    #Time
    #0.000083  0.000083
    #0.000167  0.000083

    df_tmp.rename(columns={df_tmp.columns[0]: 'Diff'}, inplace=True, errors="raise")
    #df_tmp.rename(index={0: "Diff"}, inplace=True, errors="raise")  # KeyError: '[1] not found in axis'

    # Calculate the mode for column 'Diff' and get the maximum value as mode_max.
    #print(df_tmp['Diff'].mode())
    mode_median = df_tmp['Diff'].mode().median()
    mode_max = df_tmp['Diff'].mode().max()
    median = df_tmp['Diff'].median()
    if verbose: print("Col 'diff' has a mode().median() of: ", mode_median)    # 1 days 00:00:00
    if verbose: print("Col 'diff' has a mode().max() of: ", mode_max)    # 1 days 00:00:00
    if verbose: print("Col 'diff' has a median of: ", median)      # 1 days 00:00:00
    bool_mask = mode_max

    if isinstance(df.index, pd.DatetimeIndex):
        # Dataframe index of type datetime
        idx_freq_val, idx_freq_unit = get_df_index_freq_val_unit(df=df)
        if idx_freq_unit == "D" and idx_freq_val > 28:
            # monthly interval of ~ 31 days or 365.  This requires special processing.
            # Create a boolean mask based on column 'Diff' values and mode_median
            if verbose: print("Using bool_mask of mode_median of ", mode_median)
            bool_mask = mode_median

        # Create a boolean mask based on column 'Diff' values and bool_mask
        # in order find gaps.
        #if verbose: print(df_tmp['Diff'])
        df_tmp['bool_mask'] = df_tmp['Diff'].gt(bool_mask)
        #if verbose: print(df_tmp['bool_mask'].to_string())
        count_of_true = df_tmp.loc[df_tmp['bool_mask'] == True, 'bool_mask'].count()
        if verbose: print(str(count_of_true) + " gaps found in the index for threshold of " + str(bool_mask))
        del df_tmp
        if count_of_true > 0:
            return True
        else:
            return False


    else:
        # Dataframe index of type numeric
        if verbose: print("Col 'diff' has a min of: ", df_tmp['Diff'].min()) 
        if verbose: print("Col 'diff' has a max of: ", df_tmp['Diff'].max()) 
        df_tmp['delta'] = df_tmp['Diff'] - mode_median
        if verbose: print("Col 'delta' min: ", df_tmp['delta'].min())
        if verbose: print("Col 'delta' min: ", df_tmp['delta'].max())
        if verbose: print("Col 'delta' median: ", df_tmp['delta'].median())
        #import numpy as np
        #np.isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)
        if df_tmp['delta'].median() == 0.0 or math.isclose(df_tmp['delta'].median(), 0.0):
            #print("median of col diff - mode_median values for the numeric index are nearly zero")
            del df_tmp
            return False
        bool_mask = mode_median
        df_tmp['bool_mask'] = df_tmp['Diff'].gt(mode_max)
        count_of_true = df_tmp.loc[df_tmp['bool_mask'] == True, 'bool_mask'].count()
        if verbose: print(str(count_of_true) + " gaps found in the index")
        if count_of_true > 0:
            return True
        else:
            return False
        
        

    # count occurrences of mode_max in column 'Diff'
    count_of_mode_max = df_tmp['Diff'].value_counts()[mode_max]
    count_of_gaps = df_tmp.shape[0]-count_of_mode_max
    # OR: if not count_of_mode_max == df_tmp.shape[0]:
    if count_of_gaps > 0:
        if verbose: print(str(df_tmp.shape[0]-count_of_mode_max) + " gaps found in the index")
        return True
    else:
        pass
        return False

    # count all values in name  with relative frequencies
    ds_rel_freq = df_tmp['Diff'].value_counts(normalize=True)
    #print(ds_rel_freq.to_string())
    #Diff
    #1 days    0.784756
    #3 days    0.179281
    #4 days    0.026838
    #...
    if len(ds_rel_freq) > 1:
        if verbose: print("ds_rel_freq has found gaps in the index")
        return True
    else:
        return False

    # Not the best method for a numeric index.  Use one of the prior methods.
    # Create a new column 'bool_mask' with the bool result of existance of mode_max in column 'Diff'
    if verbose: print("mode_max: ", mode_max)
    df_tmp['bool_mask'] = df_tmp['Diff'] == mode_max   
    # Count of values in column 'bool_mask' with value of False
    #print("Count of values in column 'bool_mask' with value of False: ", df_tmp.loc[df_tmp.bool_mask == False, 'bool_mask'].count())
    #print("Count of values in column 'bool_mask' with value of False: ", df_tmp['bool_mask'].value_counts()[False])
    count_of_false = df_tmp.loc[df_tmp.bool_mask == False, 'bool_mask'].count()
    count_of_false = df_tmp['bool_mask'].value_counts()[False]
    #print(df_tmp.to_string())
    if count_of_false > 0:
        if verbose: print(str(df_tmp.shape[0]-count_of_false) + " gaps found in the index")
        return True
    else:
        return False


def df_has_nan_or_nat(df=None, results_by_col=False, verbose=False):
    # By default (results_by_col=False), returns True if any column in 
    # Pandas DataFrame has NaN / NaT.  
    # If results_by_col=True, returns column index, column name, True/False if has NaN/NaT, count of NaN/Nat

    import pandas as pd
    from darts import TimeSeries

    if not isinstance(df, pd.DataFrame) and not isinstance(df, pd.Series): raise Exception("Value Error: df must be a Pandas dataframe or series")

    if results_by_col==False:
        if verbose:
            # print the rows that have NaN/NaT
            print("Rows with NaN/NaT:")
            print(df[df.isnull().any(axis=1)])
        return df.isnull().sum().any()

    has_nan_nat = False
    results = []
    for idx, colname in enumerate(df.columns):
        ds = df.iloc[:,idx]
        #if ds.hasnans and verbose: print(idx, colname, ds.hasnans, ds.isnull().sum())
        if ds.hasnans: has_nan_nat = True
        results.append([idx, colname, ds.hasnans, ds.isnull().sum()])

    if verbose:
        # print the rows that have NaN/NaT
        print(df[df.isnull().any(axis=1)])

    return tuple(results)
    


# ---------------------------------------------------------------------------

def reset_matplotlib():
    # Reset the colors to default for v2.0 (category10 color palette used by Vega and d3 originally developed at Tableau)
    import matplotlib as mpl
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])


def plt_darts_ts_stacked(ts=None, title="", labels=None, verbose=False):
    # Creates a stacked line plot with aligned X and Y axis ranges. 
    # ts can be a list of TimeSeries, a multivariate TimeSeries, or a Pandas Dataframe / Series.

    from darts import TimeSeries
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    if isinstance(ts, list): 
        if not isinstance(labels,list): raise Exception("Value Error: labels must be passed as a list of chart labels of type string")
    else:
        # Validate ts is a series, convert from Pandas Dataframe/Series if necessary
        if isinstance(ts, pd.DataFrame):
            if verbose: print("Pandas DataFrame")
            ts = TimeSeries.from_dataframe(ts)
        elif isinstance(ts, pd.Series):
            if verbose: print("Pandas Series")
            ts = TimeSeries.from_series(pd_series=ts)
        elif isinstance(ts, TimeSeries):
            #print("Darts series")
            if ts.width > 1:
                if verbose: print("Darts multivariate TimeSeries with " + str(ts.width) + " cols x " + str(len(ts)) + " rows")
            else:
                if verbose: print("Darts univariate TimeSeries with " + str(len(ts)) + " rows")
        else:
            raise Exception("Unknown type of series ", type(ts))

    reset_matplotlib()

    def plot_to_ax(ts, fig, ax, ax_idx, show_legend=False, title=''):
        if isinstance(ax, np.ndarray): 
            len_ax = len(ax)
        else:
            len_ax = 1
        if ts.width == 1 and len_ax == 1:
            ts.plot(ax=ax)
            if show_legend == False: ax.legend_ = None
        elif ts.width == 1 and len_ax > 1:
            ts.plot(ax=ax[ax_idx])
            ax[ax_idx].title.set_text(title)
            ax[ax_idx].xaxis.set_label_text('')
            if show_legend == False: ax[ax_idx].legend_ = None
        else:
            # ts.width > 1 (multiple columns/components)
            # Create individual series for plotting from each multivariate series
            df = ts.pd_dataframe()
            i = ax_idx
            for col in df.columns:
                ds = df[col]
                ds.plot(ax=ax[i])
                ax[i].title.set_text(ds.name)
                ax[i].xaxis.set_label_text('')
                if show_legend == False: ax[i].legend_ = None
                i += 1

    if not isinstance(ts, list):
        # ts is not a list
        fig, ax = plt.subplots(ts.width, figsize=(14,6), sharex=True, sharey=True)
        fig.suptitle(title)
        plot_to_ax(ts, fig, ax, ax_idx=0)
        plt.tight_layout()
    else:
        # ts is a list
        num_plots = 0
        for series in ts:
            num_plots += series.width
        fig, ax = plt.subplots(num_plots, figsize=(14,6), sharex=True, sharey=True)
        fig.suptitle(title)
        i = 0
        ax_idx = 0
        for series in ts:
            if isinstance(labels, list):
                plot_to_ax(series, fig, ax, ax_idx, title=labels[i])
            else:
                plot_to_ax(series, fig, ax, ax_idx)
            ax_idx += series.width
            i += 1
        plt.tight_layout()

    plt.show()


def plt_model_training(train=None, val_series=None, test=None, pred_input=None, pred_steps=None, pred=None, title="", past_covariates=None, future_covariates=None):
    # Generates line plot of Darts TimeSeries train, val_series, test, pred_input, the prediction interval pred_steps, past_covariates, and future_covariates
    # that are typically used for training a model.  
    # Handles univariate and multivariate series. 
    # The following TimeSeries are optional and may be None:  val_series, pred_input, pred, past_covariates, future_covariates.
    # Plots aligned X and Y axis ranges.

    # Colors:
    #   C0  train
    #   C1  val
    #   C2  test
    #   C3  pred
    #   C4  past/future covariates

    import matplotlib.pyplot as plt
    from darts import TimeSeries

    # Validate any series passed
    if train is None or not isinstance(train, TimeSeries): raise Exception("Value Error: train is not a Darts TimeSeries")
    if not val_series is None and not isinstance(val_series, TimeSeries): raise Exception("Value Error: val_series is not a Darts TimeSeries")
    if test is None or not isinstance(test, TimeSeries): raise Exception("Value Error: test is not a Darts TimeSeries")
    if not pred_input is None and not isinstance(pred_input, TimeSeries): raise Exception("Value Error: pred_input is not a Darts TimeSeries")
    if pred_steps is None or not isinstance(pred_steps, int): raise Exception("Value Error: pred_steps is not an integer")

    reset_matplotlib()

    if train.width == 1:
        fig, ax = plt.subplots(2, figsize=(14,6), sharex=True, sharey=True)
        fig.suptitle(title)

        # top chart
        if not train is None: train.plot(ax=ax[0], label="train", color='C0')
        if not val_series is None: val_series.plot(ax=ax[0], label="val_series", color='C1')
        if not test is None: test.plot(ax=ax[0], label="test", color='C2') 
        if not past_covariates is None: past_covariates.plot(ax=ax[0], label="past_covariates", linewidth=1.0, color="C4")
        if not future_covariates is None: future_covariates.plot(ax=ax[0], label="future_covariates", linewidth=1.0, linestyle='--', color="C4")

        # bottom chart
        if not test is None: 
            if pred_input is None and not val_series is None: val_series.plot(ax=ax[1], linewidth=2.5, linestyle='--', label="ground truth", color='tab:gray')
            test.plot(ax=ax[1], linewidth=2.5, linestyle='--', label="ground truth", color='tab:gray')

        if not pred_input is None and pred_input.time_index.min() <= train.time_index.max():
            pred_input.plot(ax=ax[1], label="pred_input", linewidth=1.0, color='C0')
        elif not pred_input is None:
            pred_input.plot(ax=ax[1], label="pred_input", linewidth=1.0, color='C2')
        
        if pred is None:
            test[-pred_steps:].plot(ax=ax[1], linewidth=1.0, label="prediction (simulation)", color='C3')   # simulate prediction
        else: 
            pred.plot(ax=ax[1], linewidth=1.0, label="prediction", color='C3')   
        
        ax[0].xaxis.set_label_text('')
        ax[1].xaxis.set_label_text('')
    else:
        # ts.width > 1 (multiple columns/components)

        num_plots = train.width
        if not past_covariates is None: num_plots += 1
        if not future_covariates is None: num_plots += 1

        fig, ax = plt.subplots(num_plots, figsize=(14,6), sharex=True, sharey=True, layout='constrained')
        fig.suptitle(title + "\n")      # The '\n' add additional space for the legend
        labels = []

        # Create individual series for plotting from each multivariate series
        df = train.pd_dataframe()
        i = 0
        for col in df.columns:
            ds = df[col]
            ds.plot(ax=ax[i], label="train", color='C0')
            ax[i].title.set_text(ds.name)
            ax[i].xaxis.set_label_text('')
            i += 1
        labels.append('train')

        if not val_series is None:
            df = val_series.pd_dataframe()
            i = 0
            for col in df.columns:
                ds = df[col]
                ds.plot(ax=ax[i], label="val_series", color='C1')
                ax[i].title.set_text(ds.name)
                ax[i].xaxis.set_label_text('')
                i += 1
            labels.append('val_series')

        df = test.pd_dataframe()
        i = 0
        for col in df.columns:
            ds = df[col]
            ds.plot(ax=ax[i], linewidth=2.5, linestyle='--', label="ground truth", color='tab:gray')    
            ax[i].title.set_text(ds.name)
            ax[i].xaxis.set_label_text('')
            i += 1
        labels.append('ground_truth')

        if pred_input is None and not val_series is None:
            df = val_series.pd_dataframe()
            i = 0
            for col in df.columns:
                ds = df[col]
                ds.plot(ax=ax[i], linewidth=2.5, linestyle='--', label="ground truth", color='tab:gray')
                ax[i].title.set_text(ds.name)
                ax[i].xaxis.set_label_text('')
                i += 1

        if not pred_input is None:
            df = pred_input.pd_dataframe()
            i = 0
            for col in df.columns:
                ds = df[col]
                #ds.plot(ax=ax[i], label="pred_input", color='C1')
                if pred_input.time_index.min() <= train.time_index.max():
                    ds.plot(ax=ax[i], label="pred_input", color='C0')
                else:
                    ds.plot(ax=ax[i], label="pred_input", color='C2')
                ax[i].title.set_text(ds.name)
                ax[i].xaxis.set_label_text('')
                i += 1
            labels.append('pred_input')

        if pred is None:
            df = test[-pred_steps:].pd_dataframe()
            i = 0
            for col in df.columns:
                ds = df[col]
                ds.plot(ax=ax[i], linewidth=1.0, label="prediction (simulation)", color='C3')   # simulate prediction
                ax[i].title.set_text(ds.name)
                ax[i].xaxis.set_label_text('')
                i += 1
            labels.append('pred (simulation)')
        else: 
            df = pred.pd_dataframe()
            i = 0
            for col in df.columns:
                ds = df[col]
                ds.plot(ax=ax[i], linewidth=1.0, label="prediction", color='C3')   
                ax[i].title.set_text(ds.name)
                ax[i].xaxis.set_label_text('')
                i += 1
            labels.append('pred')

        i = num_plots
        if not past_covariates is None: i -= 1
        if not future_covariates is None: i -= 1
        
        if not past_covariates is None: 
            df = past_covariates.pd_dataframe()
            for col in df.columns:
                ds = df[col]
                ds.plot(ax=ax[i], label="past_covariates", linewidth=1.0, color="C4")
                ax[i].title.set_text("past covariates")
                ax[i].xaxis.set_label_text('')
                ax[i].legend_ = None
            i += 1

        if not future_covariates is None: 
            df = future_covariates.pd_dataframe()
            for col in df.columns:
                ds = df[col]
                ds.plot(ax=ax[i], label="future_covariates", linewidth=1.0, color="C4")
                ax[i].title.set_text("future covariates")
                ax[i].xaxis.set_label_text('')
                ax[i].legend_ = None
            i += 1

        #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize='small')
        fig.legend(loc='outside upper right', fontsize='small',labels=labels)

    plt.tight_layout()
    plt.show()


def plt_model_trained(ts=None, past_covariates=None, future_covariates=None, pred_input=None, pred=None, title=""):
    # Plots a trained model .predict() resuls with pred_input and pred plotted over the original series ts.
    # If ts is multivariate, plots each component (column) in a separate axis.
    # If past or future covariates are passed, they are plotted on an individual axis. 
    # Use plt_model_training for plotting a model being trained.  

    import matplotlib.pyplot as plt
    from darts import TimeSeries

    # Validate any series passed
    if not ts is None and  not isinstance(ts, TimeSeries): raise Exception("Value Error: ts is not a Darts TimeSeries")
    if not past_covariates is None and not isinstance(past_covariates, TimeSeries): raise Exception("Value Error: past_covariates is not a Darts TimeSeries")
    if not future_covariates is None and not isinstance(future_covariates, TimeSeries): raise Exception("Value Error: future_covariates is not a Darts TimeSeries")
    if not pred_input is None and not isinstance(pred_input, TimeSeries): raise Exception("Value Error: pred_input is not a Darts TimeSeries")
    if not pred is None and not isinstance(pred, TimeSeries): raise Exception("Value Error: pred is not a Darts TimeSeries")
    if not isinstance(title, str): raise Exception("Value Error: title must be a string")

    reset_matplotlib()

    if ts.width == 1:
        
        num_plots = ts.width
        if not past_covariates is None: num_plots += 1
        if not future_covariates is None: num_plots += 1
        i = 0

        fig, ax = plt.subplots(num_plots, figsize=(14,6), sharex=True, sharey=True)
        fig.suptitle(title)
        
        if not ts is None: ts.plot(ax=ax[i], label="ts", linewidth=3.0, linestyle='--', color='tab:gray')        
        if not pred_input is None: pred_input.plot(ax=ax[i], label="pred_input", color='C1')       
        ax[i].xaxis.set_label_text('')
        if not pred is None: pred.plot(ax=ax[i], label="pred", color='C3')
        i += 1

        if not past_covariates is None:
            ax[i].xaxis.set_label_text('')
            past_covariates.plot(ax=ax[i], label="past_covariates", color='C4')
            ax[i].xaxis.set_label_text('')
            i += 1
        
        if not future_covariates is None: 
            future_covariates.plot(ax=ax[i], label="future_covariates", color='C4')
            ax[i].xaxis.set_label_text('')
            i += 1

    else:
        # ts.width > 1 (multiple columns/components)
        num_plots = ts.width
        if not past_covariates is None: num_plots += 1
        if not future_covariates is None: num_plots += 1

        fig, ax = plt.subplots(num_plots, figsize=(14,6), sharex=True, sharey=True, layout='constrained')
        fig.suptitle(title)
        
        # Create individual series for plotting from each multivariate series
        df = ts.pd_dataframe()
        i = 0
        for col in df.columns:
            ds = df[col]
            ds.plot(ax=ax[i], linewidth=3.0, linestyle='--', color='tab:gray', label='ground truth')
            ax[i].title.set_text('ds.name')
            ax[i].xaxis.set_label_text('')
            i += 1
        
        if not pred_input is None:
            df = pred_input.pd_dataframe()
            i = 0
            for col in df.columns:
                ds = df[col]
                ds.plot(ax=ax[i], label="pred_input", color='C1')
                ax[i].title.set_text(ds.name)
                ax[i].xaxis.set_label_text('')
                i += 1

        if not pred is None:
            df = pred.pd_dataframe()
            i = 0
            for col in df.columns:
                ds = df[col]
                ds.plot(ax=ax[i], label="pred", color='C3')
                ax[i].title.set_text(ds.name)
                ax[i].xaxis.set_label_text('')
                i += 1

        i = ts.width
        if not past_covariates is None:
            past_covariates.plot(ax=ax[i])
            ax[i].title.set_text("past_covariates")
            ax[i].xaxis.set_label_text('')
            ax[i].legend_ = None
            i += 1

        if not future_covariates is None:
            future_covariates.plot(ax=ax[i])
            ax[i].title.set_text("future_covariates")
            ax[i].xaxis.set_label_text('')
            ax[i].legend_ = None
            i += 1

        fig.legend(loc='outside upper right', fontsize='small',labels=['ground_truth','pred_input','pred'])
    
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------------
# Darts Torch Foreasting Models (TFM)
    

def get_darts_model_splits(ts=None, train=0.7, val_series=0.1, test=0.2, pred_input=0.3, pred_steps=0.4, min_max_scale=True, plot=False, verbose=True):
    # Creates splits of Darts TimeSeries ts based on the percentage values passed in the 
    # arguments train, val_series, test, pred_input, and pred_steps (n).  
    # If ts is a Pandas Dataframe or Series, it will be converted to a Darts TimeSeries.
    # Returns train, val_series, test, pred_input, pred_steps (val_series and pred_input may be None).
    
    # If min_max_scale == True, scales the other series to a range of 0.0 to +1.0
    # Note that the scaler is fit on series 'train' only to prevent data leakage to the series val_series, test, and pred_input.
    # Use series train to perform any inverse transformation.  Ex.  pred = train.inverse_transform(pred)
    # https://unit8co.github.io/darts/examples/18-TiDE-examples.html#Data-Loading-and-preparation
    # https://towardsdatascience.com/the-easiest-way-to-forecast-time-series-using-n-beats-d778fcc2ba60
    
    # train and test arguments must be greater than 0.0 and less than 1.0
    # val_series and pred_input are optional. 
    # val_series is a validation series passed as input to model.fit(series=val_series).
    # pred_input is always a portion of the series test and is assigned as: model.predict(series=pred_input)
    # pred_steps is the percentage of series test to allocate for forecasting.  It is assigned as model.predict(n=pred_steps)
    # pred_steps are always sequential to pred_input and both are a portion of the series test. 
    # The sum of the arguments train, val_series, and test must be 1.0 (100%).
    # The split of series 'ts' to train, val_series, and test will be sequential relative to the index.
    # If future covariates are employed, it is strongly recommended to allocate less than 100% of series test to pred_input and pred_steps. 
    # Preserves static covariate data from ts and allocates to each split.
    # Encodes/transformes categorical static covariates to numeric. 
    # Handles univariate and multivariate TimeSeries, but NOT multiple TimeSeries.  

    from darts import TimeSeries
    from darts.dataprocessing.transformers.scaler import Scaler
    from darts.dataprocessing.transformers import StaticCovariatesTransformer
    import pandas as pd
    import matplotlib.pyplot as plt
    
    if verbose: print("get_darts_model_splits()..")

    static_covariate_transformer = StaticCovariatesTransformer()
    scaler = Scaler()  # default uses sklearn's MinMaxScaler.  
    
    # Validate the inputs
    if not isinstance(ts, TimeSeries): raise Exception("Value Error: argument ts is not a Darts TimeSeries")
    if not isinstance(train, float): raise Exception("Value Error: argument train is not a float")
    if not val_series is None and not isinstance(val_series, float): raise Exception("Value Error: argument val_series is not a float")
    if not isinstance(test, float): raise Exception("Value Error: argument test is not a float")
    if not pred_input is None and not isinstance(pred_input, float): raise Exception("Value Error: argument pred_input is not a float")
    if not isinstance(pred_steps, float): raise Exception("Value Error: argument pred_steps is not a float")
    if not isinstance(min_max_scale, bool): raise Exception("Value Error: argument min_max_scale is not a bool value")
    if not isinstance(plot, bool): raise Exception("Value Error: argument plot is not a bool value")
    if not isinstance(verbose, bool): raise Exception("Value Error: argument verbose is not a bool value")

    # Convert any arguments with value None to 0
    train = 0 if not train else train
    val_series = 0 if not val_series else val_series
    test = 0 if not test else test
    pred_input = 0 if not pred_input else pred_input
    pred_steps = 0 if not pred_steps else pred_steps

    if train > 1.0 or train <= 0.0: raise Exception("Value Error: train must be between 0.0 and +1.0")
    if val_series > 1.0 or val_series < 0.0: raise Exception("Value Error: val_series must be between 0.0 and +1.0")
    if test > 1.0 or test <= 0.0: raise Exception("Value Error: test must be between 0.0 and +1.0")
    if pred_input > 1.0 or pred_input < 0.0: raise Exception("Value Error: pred_input must be between 0.0 and +1.0 (a fractional portion of test)")
    if pred_steps > 1.0 or pred_steps <= 0.0: raise Exception("Value Error: pred_steps must be between 0.0 and +1.0 (a fractional portion of test)")
    if not train+val_series+test==1.0: raise Exception("Value Error: train+test != 1.0", sum(train,val_series,test))

    #if train == 0 or test == 0: raise Exception("Value Error: arguments train and test passed to get_darts_model_splits() must be greater than 0.0 and less than 1.0")

    # Validate ts is a series, convert if necessary
    if isinstance(ts, pd.DataFrame):
        if verbose: print("Pandas DataFrame")
        ts = TimeSeries.from_dataframe(ts)
    elif isinstance(ts, pd.Series):
        if verbose: print("Pandas Series")
        ts = TimeSeries.from_series(pd_series=ts)
    elif isinstance(ts, TimeSeries):
        #print("Darts series")
        if ts.width > 1:
            if verbose: print("Darts multivariate TimeSeries with " + str(ts.width) + " cols x " + str(len(ts)) + " rows")
        else:
            if verbose: print("Darts univariate TimeSeries with " + str(len(ts)) + " rows")
    else:
        raise Exception("Unknown type of series ", type(ts))

    template = "{0:>20}|{1:>25}|{2:>25}|{3:>15}|{4:>15}" # column widths: 15, 20, 20, 15    '>' right justify
    if verbose: print(template.format("series", "index start", "index end", "length", "width")) # header
    if verbose: print(template.format("ts", str(ts.time_index.min()), str(ts.time_index.max()), len(ts), ts.width))

    # Split ts into train, val_series, test, & pred_input
    len_train = int(len(ts)*train)
    len_val_series = int(len(ts)*val_series)
    len_test = int(len(ts) - len_val_series - len_train)
    len_pred_input = int(len_test*pred_input)
    if not (len_train+len_test+len_val_series) == len(ts): raise Exception("ERROR: sum of train, val_series, test != len(ts) !!!")
    
    train = ts[:len_train]
    if len_val_series == 0:
        val_series = None
        test = ts[len_train:len_train+len_test]
    else:
        val_series = ts[len_train:len_train+len_val_series]
        test = ts[len_train+len_val_series:len_train+len_val_series+len_test]

    if len_pred_input == 0:
        pred_input = None
    else:
        pred_input = test[:len_pred_input]

    # Scale all series returned to +/-1.0 if requested
    if min_max_scale == True: 
        train = scaler.fit_transform(train)
        if not val_series is None: val_series = scaler.transform(val_series)
        if not pred_input is None: pred_input = scaler.transform(pred_input)
        test = scaler.transform(test)
    else:
        scaler = None

    # Add any static covariates in ts to the split series
    if ts.has_static_covariates:
        # encode/transform categorical static covariates (required by models such as DLinear, ...)
        ts = static_covariate_transformer.fit_transform(ts)
        train = train.with_static_covariates(ts.static_covariates)
        if not val_series is None: val_series = val_series.with_static_covariates(ts.static_covariates)
        if not pred_input is None: pred_input = pred_input.with_static_covariates(ts.static_covariates)
        test = test.with_static_covariates(ts.static_covariates)

    if verbose: print(template.format("train", str(train.time_index.min()), str(train.time_index.max()), len(train), train.width))
    if not val_series is None and verbose == True: print(template.format("val_series", str(val_series.time_index.min()), str(val_series.time_index.max()), len_val_series, val_series.width))
    if verbose: print(template.format("test", str(test.time_index.min()), str(test.time_index.max()), len(test), test.width))
    if not pred_input is None and verbose == True: print(template.format("pred_input", str(pred_input.time_index.min()), str(pred_input.time_index.max()), len(pred_input), pred_input.width))

    pred_steps = int(len(test)*pred_steps)
    if not pred_input is None and len(pred_input)+pred_steps > len(test):
        pred_steps = len(test) - len(pred_input)
        print("WARNING: pred_steps + len(pred_input) > len(test) .. adjusting pred_steps to " + str(pred_steps) + " rows")
        print("WARNING: if past covariates are used with PyTorch (Lightning)-based Models, then len(pred_input) + pred_steps must be must less than len(test) !")

    if pred_input is None:
        #if verbose: print("pred_steps: ", test[:pred_steps].time_index.min(), " ..", test[:pred_steps].time_index.max(), "\tlength: ", len(test[:pred_steps].time_index))  
        if verbose: print(template.format("pred_steps", str(test[:pred_steps].time_index.min()), str(test[:pred_steps].time_index.max()), len(test[:pred_steps]), test[:pred_steps].width))
    else:
        #if verbose: print("pred_steps: ", test[len(pred_input):len(pred_input)+pred_steps:].time_index.min(), " ..", test[len(pred_input):len(pred_input)+pred_steps:].time_index.max(), "\tlength: ", len(test[len(pred_input):len(pred_input)+pred_steps:].time_index))  
        if verbose: print(template.format("pred_steps", str(test[len(pred_input):len(pred_input)+pred_steps:].time_index.min()), str(test[len(pred_input):len(pred_input)+pred_steps:].time_index.max()), len(test[len(pred_input):len(pred_input)+pred_steps:]), test[len(pred_input):len(pred_input)+pred_steps:].width))

    if len_pred_input+pred_steps == len(test):
        if verbose: print("WARNING: If you have future covariates assigned to the model .fit() & .predict() methods, you will need spare time span after the end of pred_steps (n).  Reduce pred_input and/or pred_steps so their total is less than 1.0")
    elif len_pred_input+pred_steps < len(test):
        if verbose: print(template.format("future covariates", str(test[len_pred_input+pred_steps:len(test):].time_index.min()), str(test[len_pred_input+pred_steps:len(test):].time_index.max()), len(test)-len_pred_input-pred_steps, test.width))

    if plot:
        reset_matplotlib()
        fig, ax = plt.subplots(1, figsize=(14,6), sharex=True, sharey=True)
        fig.suptitle("get_darts_model_splits()")
        train.plot(ax=ax, label="train", color="C0")
        if not val_series is None: val_series.plot(ax=ax, label="val", color="C1")
        if pred_input is None:
            test.plot(ax=ax, label="test", linewidth=3.0, linestyle='--', color="C2")
            test[:pred_steps].plot(ax=ax, label="pred_steps", linewidth=1.0, color="C3")
        else:
            test.plot(ax=ax, label="test", linewidth=3.0, linestyle='--', color="C2")
            pred_input.plot(ax=ax, label="pred_input", linewidth=1.0, color="C3")
            test[len(pred_input):len(pred_input)+pred_steps:].plot(ax=ax, label="pred_steps", linewidth=1.0, color="C2")
        ax.xaxis.set_label_text('')
        ax.yaxis.set_label_text('')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return train, val_series, test, pred_input, pred_steps


def get_darts_tfm_arguments(model_name=None, train=None, val_series=None, test=None, pred_input=None, pred_steps=None, verbose=True):
    # Returns recommended (safe but not optimized) input_chunk_length and output_chunk length for PyTorch (Lightning)-based Models (Torch Forecasting Models (TFMs)),
    # as well as  input_chunk_length_max, output_chunk_length_max, sum_chunk_length_limit based on the train/val_series/test/pred_input series
    # and value for pred_steps (n).  
    # Performs extensive checks and prints useful information if verbose=True.
    # Utilizes the modeling split methodology adopted / expected by get_darts_model_splits(), but get_darts_model_splits() is not required.
    # Use the returned values input_chunk_length_max, output_chunk_length_max, sum_chunk_length_limit for optimization of input_chunk_length and output_chunk length.
    # If the model is RNNModel, then output_chunk_length=1.
    # If the model is TCNModel, then the output_chunk_length must be smaller than the input_chunk_length

    # RNNModel(input_chunk_length, model='RNN', hidden_dim=25, n_rnn_layers=1, dropout=0.0, training_length=24, **kwargs)
    # RNNModels accept a training_length parameter at model creation instead of output_chunk_length. 
    # Internally the output_chunk_length for these models is automatically set to 1. 
    # For training, past target must have a minimum length of training_length + 1 and for prediction, a length of input_chunk_length.
    # Accepts output_chunk_length as argument and doesn't throw error, but the value is ignored. 
    
    # BlockRNNModel(input_chunk_length, output_chunk_length, model='RNN', hidden_dim=25, n_rnn_layers=1, hidden_fc_sizes=None, dropout=0.0, **kwargs)
    # NBEATSModel(input_chunk_length, output_chunk_length, generic_architecture=True, num_stacks=30, num_blocks=1, num_layers=4, layer_widths=256, expansion_coefficient_dim=5, trend_polynomial_degree=2, dropout=0.0, activation='ReLU', **kwargs)[source]
    # NHiTSModel(input_chunk_length, output_chunk_length, num_stacks=3, num_blocks=1, num_layers=2, layer_widths=512, pooling_kernel_sizes=None, n_freq_downsample=None, dropout=0.1, activation='ReLU', MaxPool1d=True, **kwargs)
    # TCNModel(input_chunk_length, output_chunk_length, kernel_size=3, num_filters=3, num_layers=None, dilation_base=2, weight_norm=False, dropout=0.2, **kwargs)
    # TransformerModel(input_chunk_length, output_chunk_length, d_model=64, nhead=4, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=512, dropout=0.1, activation='relu', norm_type=None, custom_encoder=None, custom_decoder=None, **kwargs)
    # TFTModel(input_chunk_length, output_chunk_length, hidden_size=16, lstm_layers=1, num_attention_heads=4, full_attention=False, feed_forward='GatedResidualNetwork', dropout=0.1, hidden_continuous_size=8, categorical_embedding_sizes=None, add_relative_index=False, loss_fn=None, likelihood=None, norm_type='LayerNorm', use_static_covariates=True, **kwargs)
    # DLinearModel(input_chunk_length, output_chunk_length, shared_weights=False, kernel_size=25, const_init=True, use_static_covariates=True, **kwargs)
    # NLinearModel(input_chunk_length, output_chunk_length, shared_weights=False, const_init=True, normalize=False, use_static_covariates=True, **kwargs)
    # TiDEModel(input_chunk_length, output_chunk_length, num_encoder_layers=1, num_decoder_layers=1, decoder_output_dim=16, hidden_size=128, temporal_width_past=4, temporal_width_future=4, temporal_decoder_hidden=32, use_layer_norm=False, dropout=0.1, use_static_covariates=True, **kwargs)

    devmt = False

    if verbose: print("\n", "get_darts_tfm_arguments()")

    from darts import TimeSeries

    if not isinstance(model_name, str): raise Exception("Value Error: model_name must be string type with a value matching a Darts TFM name")
    tft_models = ('RNNModel','BlockRNNModel','NBEATSModel','NHiTSModel','TCNModel','TransformerModel','TFTModel','DLinearModel','NLinearModel','TiDEModel')
    if not model_name in tft_models: raise Exception("Value Error: model_name must be string type with a value matching a Darts TFM name", model_name)

    # Validate any series passed
    if train is None or not isinstance(train, TimeSeries): raise Exception("Value Error: train is not a Darts TimeSeries")
    if not val_series is None and not isinstance(val_series, TimeSeries): raise Exception("Value Error: val_series is not a Darts TimeSeries")
    if test is None or not isinstance(test, TimeSeries): raise Exception("Value Error: test is not a Darts TimeSeries")
    if not pred_input is None and not isinstance(pred_input, TimeSeries): raise Exception("Value Error: pred_input is not a Darts TimeSeries")
    if pred_steps is None or not isinstance(pred_steps, int): raise Exception("Value Error: pred_steps is not None or an integer")
    if not isinstance(verbose, bool): raise Exception("Value Error: argument verbose is not a bool value")

    # Any PyTorch model except RNNModel requires input_chunk_length and output_chunk_length
    # The TCNModel requires: output_chunk_length < input_chunk_length < kernel_size

    # Define input_chunk_length & output_chunk_length
    # NOTE: for Torch Forecasting Models except RNNModel: 
    #   input_chunk_length + output_chunk_length < len(train) OR < len(pred_input)    

    sum_chunk_length_limit = len(train)
    training_length_max = None
    model_specific = {}
    if val_series is None and not pred_input is None:
        if devmt==True: print("val_series is None and not pred_input is None")
        sum_chunk_length_limit = len(pred_input)+1
        input_chunk_length_max = len(pred_input)
        if model_name == 'RNNModel':
            output_chunk_length_max = 1
            training_length_max = len(train)-1
        else:
            output_chunk_length_max = len(pred_input)
    elif val_series is None and pred_input is None:
        if devmt==True: print("val_series is None and pred_input is None")
        # train will be used for .predict(series=train) rather than .predict(series=pred_input)
        if model_name == 'RNNModel':
            output_chunk_length_max = 1
            input_chunk_length_max = len(train)
            sum_chunk_length_limit = len(train)+1
            training_length_max = len(train)-1
        else:
            output_chunk_length_max = len(train)-1
            input_chunk_length_max = len(train)-1
            sum_chunk_length_limit = len(train)
    elif not val_series is None and not pred_input is None:
        if devmt==True: print("not val_series is None and not pred_input is None")
        if len(val_series) >= len(pred_input):
            if devmt: print("len(val_series) >= len(pred_input)")
            input_chunk_length_max = len(pred_input)
            if model_name == 'RNNModel':
                output_chunk_length_max = 1
                sum_chunk_length_limit = len(pred_input)+1
                training_length_max = len(val_series)-1
            else:
                output_chunk_length_max = len(pred_input)-1
                sum_chunk_length_limit = len(val_series)
        else:
            # len(val_series) < len(pred_input)
            if devmt: print("len(val_series) < len(pred_input)")
            if model_name == 'RNNModel':
                output_chunk_length_max = 1
                input_chunk_length_max = len(pred_input)
                sum_chunk_length_limit = len(pred_input)+1
                training_length_max = len(val_series)-1
            else:
                output_chunk_length_max = len(val_series)-1
                input_chunk_length_max = len(val_series)-1
                sum_chunk_length_limit = len(val_series)
    elif not val_series is None and pred_input is None:
        if devmt==True: print("not val_series is None and pred_input is None")
        if model_name == 'RNNModel':
            output_chunk_length_max = 1
            input_chunk_length_max = len(train)
            sum_chunk_length_limit = len(train)+1
            training_length_max = len(val_series)-1
        else:
            output_chunk_length_max = len(val_series)-1
            input_chunk_length_max = len(val_series)-1
    else:
        raise Exception("Logic Error: unanticipated logic situation!")

    if verbose: print("input_chunk_length_max:" , input_chunk_length_max)

    len_in = input_chunk_length_max//2 
    if model_name == 'RNNModel':
        print("RNNModels accept a training_length parameter at model creation instead of output_chunk_length.")
        print("Internally the output_chunk_length for these models is automatically set to 1 (you can pass it as an argument, but it will be ignored).")
        #print("For training, past target must have a minimum length of training_length + 1 and for prediction, a length of input_chunk_length.")
        if verbose: print("Try input_chunk_length: " + str(1) + " .. " + str(input_chunk_length_max))
        if verbose: print("training_length_max:",training_length_max)
        if devmt: print("sum_chunk_length_limit:", sum_chunk_length_limit)
        len_out = 1
        model_specific['training_length_max'] = training_length_max
    elif model_name == 'TCNModel':
        # *** TCNModel: The output length must be strictly smaller than the input length
        len_out = output_chunk_length_max//2-1
        if len_out >= pred_steps and pred_steps//2 < output_chunk_length_max: len_out = pred_steps//2
        while len_out > len_in:
            len_out -= 1
        if len_out < 1: len_out = 1
    else:
        if verbose: print("output_chunk_length_max:" , output_chunk_length_max, "\trecommended <= ", pred_steps, " to allow auto-regression for Torch Forecasting Models")
        if verbose: print("Try input_chunk_length " + str(1) + " .. " + str(input_chunk_length_max) + " & output_chunk_length " + str(1) + " .. " + str(output_chunk_length_max), "  AND sum(input_chunk_length+output_chunk_length) <= " + str(sum_chunk_length_limit))
        len_out = output_chunk_length_max//2
        if len_out >= pred_steps and pred_steps//2 < output_chunk_length_max: len_out = pred_steps//2
       
    if verbose: print("Returning: input_chunk_length:", len_in, "\t\toutput_chunk_length:", len_out, "\t\tsum(input_chunk_length+output_chunk_length):", len_in+len_out)

    if devmt==True: 
        if verbose: print("\n")
        return len_in, len_out, input_chunk_length_max, output_chunk_length_max, sum_chunk_length_limit, model_specific

    if (len_in+len_out) > sum_chunk_length_limit: print("WARNING: sum of input_chunk_length & output_chunk_length must be <= sum_chunk_length_limit of " + str(sum_chunk_length_limit))
    if (len_in+len_out) > sum_chunk_length_limit: raise Exception("ERROR: sum of input_chunk_length & output_chunk_length must be <= sum_chunk_length_limit of " + str(sum_chunk_length_limit))
    if len_in > input_chunk_length_max: raise Exception("ERROR: input_chunk_length > " + str(input_chunk_length_max))
    if len_out > output_chunk_length_max: raise Exception("ERROR: output_chunk_length > " + str(output_chunk_length_max))
    # Torch Forecasting Models, Setting n <= output_chunk_length prevents auto-regression
    if pred_steps <= len_out: print("WARNING: Setting n <= output_chunk_length prevents auto-regression for Torch Forecasting Models\t", pred_steps, "<=", len_out)
    if verbose: print("\n")

    return len_in, len_out, input_chunk_length_max, output_chunk_length_max, sum_chunk_length_limit, model_specific


def darts_train_fit_pred_tfm(ts, train, val_series, test, pred_input, pred_steps, 
                    model_class, min_max_scale, metric,
                    input_chunk_length, output_chunk_length, 
                    past_covariates, val_past_covariates, future_covariates, val_future_covariates, 
                    verbose=True):
    # Train, fit, predict a Darts Torch Forecasting Model (TFM)
    # and calculate the model performance measured by metric.
    # Returns the result of the metric calculation, the prediction, 
    # and the model as: metric_result, pred, model.
    # Called by get_darts_tfm_arguments_optimized().
    # Series ts is used for reference by slice_tfm_covariates_for_model_training().

    from darts import TimeSeries

    # Valiate the inputs
    if model_class is None: raise Exception("Value Error: you must pass a Darts Torch Forecasting Model for the argument model_class")
    from darts.models import (RNNModel,BlockRNNModel,NBEATSModel,NHiTSModel,TCNModel,TransformerModel,TFTModel,DLinearModel,NLinearModel,TiDEModel)
    tft_models = (RNNModel,BlockRNNModel,NBEATSModel,NHiTSModel,TCNModel,TransformerModel,TFTModel,DLinearModel,NLinearModel,TiDEModel)
    if not model_class in tft_models: raise Exception ("Value Error: argument model_class is not a Torch Forecasting Model. ", str(model_class))

    if metric is None: raise Exception("Value Error: argument metric must not be None.  It should be a Darts metric.  See: from darts.metrics import [metric]")
    if not isinstance(min_max_scale, bool): raise Exception("Value Error: min_max_scale must be boolean type with value of True or False")
    if not isinstance(verbose, bool): raise Exception("Value Error: verbose must be boolean type with value of True or False")
    if not isinstance(verbose, bool): raise Exception("Value Error: argument verbose is not a bool value")

    if not isinstance(input_chunk_length, int): raise Exception("Value Error: input_chunk_length must be int type >= 1")
    if not isinstance(output_chunk_length, int): raise Exception("Value Error: output_chunk_length_max must be int type >= 1")
    if input_chunk_length < 1: raise Exception("Value Error: input_chunk_length must be >= 1 ", input_chunk_length)
    if output_chunk_length < 1: raise Exception("Value Error: output_chunk_length must be >= 1 ", output_chunk_length)

    # Validate any series passed
    if train is None or not isinstance(train, TimeSeries): raise Exception("Value Error: train is not a Darts TimeSeries")
    if not val_series is None and not isinstance(val_series, TimeSeries): raise Exception("Value Error: val_series is not a Darts TimeSeries")
    if test is None or not isinstance(test, TimeSeries): raise Exception("Value Error: test is not a Darts TimeSeries")
    if not pred_input is None and not isinstance(pred_input, TimeSeries): raise Exception("Value Error: pred_input is not a Darts TimeSeries")
    if pred_steps is None or not isinstance(pred_steps, int): raise Exception("Value Error: pred_steps is not None or an integer")
    # past_covariates=None, val_past_covariates=None, future_covariates=None, val_future_covariates=None

    if val_series is None and not val_past_covariates is None: raise Exception("Value Error: val_past_series must be None if val_series is None")
    if val_series is None and not val_future_covariates is None: raise Exception("Value Error: val_future_covariates must be None if val_series is None")
    if not val_series is None and not past_covariates is None and val_past_covariates is None: raise Exception("Value Error: val_past_series must NOT be None if val_series is not None and past_covariates is not None")
    if not val_series is None and not future_covariates is None and val_future_covariates is None: raise Exception("Value Error: val_future_covariates must NOT be None if val_series is not None and future_covariates is not None")

    # past_covariates, val_past_covariates, future_covariates, val_future_covariates, static_covariates
    if not past_covariates is None and not isinstance(past_covariates, TimeSeries): raise Exception("Value Error: past_covariates is not a Darts TimeSeries")
    if not val_past_covariates is None and not isinstance(val_past_covariates, TimeSeries): raise Exception("Value Error: val_past_covariates is not a Darts TimeSeries")
    if not future_covariates is None and not isinstance(future_covariates, TimeSeries): raise Exception("Value Error: future_covariates is not a Darts TimeSeries")
    if not val_future_covariates is None and not isinstance(val_future_covariates, TimeSeries): raise Exception("Value Error: val_future_covariates is not a Darts TimeSeries")

    # Slice any covariates to the proper time span
    cv_past, cv_val_past, cv_future, cv_val_future = slice_tfm_covariates_for_model_training(ts=ts, train=train, val_series=val_series, test=test, pred_input=pred_input, pred_steps=pred_steps,
                                                                              min_max_scale=min_max_scale,
                                                                              output_chunk_length=output_chunk_length,
                                                                              past_covariates=past_covariates, val_past_covariates=val_past_covariates, future_covariates=future_covariates, val_future_covariates=val_future_covariates,
                                                                              verbose=verbose)
    if verbose: print("\ndarts_train_fit_pred_tfm()..")
    
    model = model_class(input_chunk_length=input_chunk_length, output_chunk_length=output_chunk_length)
    
    # series, past_covariates=None, future_covariates=None, val_series=None, val_past_covariates=None, val_future_covariates=None,
    model.fit(series=train, past_covariates=cv_past, future_covariates=cv_future, val_series=val_series, val_past_covariates=cv_val_past, val_future_covariates=cv_val_future)
    
    # n, series=None, past_covariates=None, future_covariates=None
    pred = model.predict(n=pred_steps, series=pred_input, past_covariates=cv_past, future_covariates=cv_future)  # defaults to series=train when pred_input==None
    
    metric_result = metric(ts.slice_intersect(pred), pred)
    
    return metric_result, pred, model


def get_darts_tfm_arguments_optimized(model_name="", model_class=None, input_chunk_length_max=None, output_chunk_length_max=None, sum_chunk_length_limit=None, ts=None, train=None, test=None, val_series=None, pred_input=None, pred_steps=None, past_covariates=None, val_past_covariates=None, future_covariates=None, val_future_covariates=None, metric=None, min_max_scale=True, coarse_only=True, verbose=True):
    # Optimizes input_chunk_length and output_chunk_length within the limits
    # of input_chunk_length_max, output_chunk_length, and sum_chunk_length_limit.
    # For the coarse optimization, a range of 1 to the maximum size in steps of 
    # powers of 2 (1,2,4,8,16,..) is employed.
    # If coarse_only=False, then a fine optimization continues and optimizes
    # between the powers of two found on either side of the coarse optimization,
    # but within the limits of input_chunk_length_max, output_chunk_length, and sum_chunk_length_limit.
    
    # Use the function get_darts_tfm_arguments() to get the following required inputs:
    #   input_chunk_length_max
    #   output_chunk_length_max
    #   sum_chunk_length_limit

    # Expects the model to be split according to the methodology supported by the
    # custom function get_darts_model_splits(). 
    
    # Does not use .gridsearch() because of the relational limitations between
    # input_chunk_length & output_chunk_length will raise errors with several models.

    from time import perf_counter
    from operator import itemgetter

    from darts import TimeSeries

    # Valiate the inputs
    if not isinstance(model_name, str): raise Exception("Value Error: model_name must be string type corresponding to the Darts model name")
    if model_class is None: raise Exception("Value Error: you must pass a Darts model for the argument model_class")
    if metric is None: raise Exception("Value Error: argument metric must not be None.  It should be a Darts metric.  See: from darts.metrics import [metric]")
    if not isinstance(min_max_scale, bool): raise Exception("Value Error: min_max_scale must be boolean type with value of True or False")
    if not isinstance(verbose, bool): raise Exception("Value Error: verbose must be boolean type with value of True or False")
    if not isinstance(input_chunk_length_max, int): raise Exception("Value Error: input_chunk_length must be int type >= 1")
    if not isinstance(output_chunk_length_max, int): raise Exception("Value Error: output_chunk_length_max must be int type >= 1")
    if not isinstance(sum_chunk_length_limit, int): raise Exception("Value Error: sum_chunk_length_limit must be int type >= 1")

    # Validate any series passed
    if train is None or not isinstance(train, TimeSeries): raise Exception("Value Error: train is not a Darts TimeSeries")
    if not val_series is None and not isinstance(val_series, TimeSeries): raise Exception("Value Error: val_series is not a Darts TimeSeries")
    if test is None or not isinstance(test, TimeSeries): raise Exception("Value Error: test is not a Darts TimeSeries")
    if not pred_input is None and not isinstance(pred_input, TimeSeries): raise Exception("Value Error: pred_input is not a Darts TimeSeries")
    if pred_steps is None or not isinstance(pred_steps, int): raise Exception("Value Error: pred_steps is not None or an integer")
    # past_covariates=None, val_past_covariates=None, future_covariates=None, val_future_covariates=None

    if val_series is None and not val_past_covariates is None: raise Exception("Value Error: val_past_series must be None if val_series is None")
    if val_series is None and not val_future_covariates is None: raise Exception("Value Error: val_future_covariates must be None if val_series is None")
    if not val_series is None and not past_covariates is None and val_past_covariates is None: raise Exception("Value Error: val_past_series must NOT be None if val_series is not None and past_covariates is not None")
    if not val_series is None and not future_covariates is None and val_future_covariates is None: raise Exception("Value Error: val_future_covariates must NOT be None if val_series is not None and future_covariates is not None")

    # past_covariates, val_past_covariates, future_covariates, val_future_covariates, static_covariates
    if not past_covariates is None and not isinstance(past_covariates, TimeSeries): raise Exception("Value Error: past_covariates is not a Darts TimeSeries")
    if not val_past_covariates is None and not isinstance(val_past_covariates, TimeSeries): raise Exception("Value Error: val_past_covariates is not a Darts TimeSeries")
    if not future_covariates is None and not isinstance(future_covariates, TimeSeries): raise Exception("Value Error: future_covariates is not a Darts TimeSeries")
    if not val_future_covariates is None and not isinstance(val_future_covariates, TimeSeries): raise Exception("Value Error: val_future_covariates is not a Darts TimeSeries")

    devmt = False        # shortens execution (for logig testing only)

    if verbose: print("\n---------------------------------------------------------------------------")
    if verbose: print("get_darts_tfm_arguments_optimized()")
    t_start = perf_counter()

    if model_name == "TCNModel":
        # Cannot use .gridsearch() for TCNModel because: 
        #   The output length must be strictly smaller than the input length
        #   The kernel size (size of every kernel in a convolutional layer) must be strictly smaller than the input length.  kererl_size = 3 by default 
        #   output_chunk_length < kernel_size < input_chunk_length
        if verbose: print("#+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-")
        if verbose: print("Coarse optimization of " + model_name + "..")
        kernel_size = 3     # default for TCNModel
        i = powers_of_two_in_range(2, input_chunk_length_max)
        i.append(input_chunk_length_max)
        # Filter by kernel_size
        i = list(filter(lambda x: kernel_size<x, i))
        o = powers_of_two_in_range(2, output_chunk_length_max)
        o.append(output_chunk_length_max)
        i.sort(reverse=True)
        o.sort(reverse=True)
        if len(i) == 0: raise Exception("Unable to find input_chunk_length between " + str(kernel_size) + " (kernel_size) and " + str(input_chunk_length_max) + "(input_chunk_length_max)")
        if verbose: print("input_chunk_length:", i)
        if verbose: print("output_chunk_length:", o)
        parameters = {'input_chunk_length': i, 'output_chunk_length': o}
        results = []
        model_args = {}
        for input_chunk_length in i:
            for output_chunk_length in o:
                if output_chunk_length < input_chunk_length and (input_chunk_length+output_chunk_length) <= sum_chunk_length_limit:
                    metric_result, pred, model = darts_train_fit_pred_tfm(ts, train, val_series, test, pred_input, pred_steps, 
                                                                        model_class, min_max_scale, metric,
                                                                        input_chunk_length=input_chunk_length, output_chunk_length=output_chunk_length, 
                                                                        past_covariates=past_covariates, val_past_covariates=val_past_covariates, future_covariates=future_covariates, val_future_covariates=val_future_covariates,
                                                                        verbose=verbose)
                    results.append([metric_result, input_chunk_length, output_chunk_length])
                    if verbose: print("input_chunk_length: ", input_chunk_length, "\toutput_chunk_length: ", output_chunk_length, "\tmetric: ", metric_result)
        if len(results) == 0: raise Exception("ERROR: No optimization results could be obtained for " + model_name + " using get_darts_tfm_arguments_optimized()")
        # Sort the results and get the one with the lowest value (best) metric
        results = sorted(results, key=itemgetter(0))
        if devmt:
            print(model_name + " input_chunk_length/output_chunk_length optimization:")
            for result in results:
                print(*result)
        model_args['input_chunk_length'] = results[0][1]
        model_args['output_chunk_length'] = results[0][2]
        if verbose: print("Coarse optimization results:")
        if verbose: print("input_chunk_length:", model_args['input_chunk_length'])
        if verbose: print("output_chunk_length:", model_args['output_chunk_length'])
        metric_result = results[0][0]

        if coarse_only == False:
            # Fine optimization of model arguments
            if verbose: print("#+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-")
            if verbose: print("Fine optimization of " + model_name + "..")
            i = list(range(model_args['input_chunk_length']-2**2, model_args['input_chunk_length']+2**2+1))
            o = list(range(model_args['output_chunk_length']-2**2, model_args['output_chunk_length']+2**2+1))
            # Filter based on input_chunk_length_max and output_chunk_length_max
            i = list(filter(lambda x: x<=input_chunk_length_max, i))
            i = list(filter(lambda x: x>0, i))
            o = list(filter(lambda x: x<=output_chunk_length_max, o))
            o = list(filter(lambda x: x>0, o))
            # Filter by kernel_size
            i = list(filter(lambda x: kernel_size<x, i))
            if len(i) == 0: i = list(range(model_args['input_chunk_length']-2**2, model_args['input_chunk_length']+1))
            if len(o) == 0: o = list(range(model_args['output_chunk_length']-2**2, model_args['output_chunk_length']+1))
            if len(i) == 0: i = [model_args['input_chunk_length']]
            if len(o) == 0: o = [model_args['output_chunk_length']]
            i = list(filter(lambda x: x<=input_chunk_length_max, i))
            i = list(filter(lambda x: x>0, i))
            o = list(filter(lambda x: x<=output_chunk_length_max, o))
            o = list(filter(lambda x: x>0, o))
            # Filter by kernel_size
            i = list(filter(lambda x: kernel_size<x, i))
            if len(i) == 0: i = [model_args['input_chunk_length']]
            if len(o) == 0: o = [model_args['output_chunk_length']]
            i.sort(reverse=True)
            o.sort(reverse=True)
            if verbose: print("input_chunk_length:", i)
            if verbose: print("output_chunk_length:", o)
            parameters = {'input_chunk_length': i, 'output_chunk_length': o}
            results = []
            model_args = {}
            for input_chunk_length in i:
                for output_chunk_length in o:
                    if output_chunk_length < input_chunk_length and (input_chunk_length+output_chunk_length) <= sum_chunk_length_limit:
                        metric_result, pred, model = darts_train_fit_pred_tfm(ts, train, val_series, test, pred_input, pred_steps, 
                                                                            model_class, min_max_scale, metric,
                                                                            input_chunk_length=input_chunk_length, output_chunk_length=output_chunk_length, 
                                                                            past_covariates=past_covariates, val_past_covariates=val_past_covariates, future_covariates=future_covariates, val_future_covariates=val_future_covariates,
                                                                            verbose=verbose)
                        results.append([metric_result, input_chunk_length, output_chunk_length])
                        if verbose: print("input_chunk_length: ", input_chunk_length, "\toutput_chunk_length: ", output_chunk_length, "\tmetric: ", metric_result)
            if len(results) == 0: raise Exception("ERROR: No optimization results could be obtained for " + model_name + " using get_darts_tfm_arguments_optimized()")
            # Sort the results and get the one with the lowest value (best) metric
            results = sorted(results, key=itemgetter(0))
            if devmt:
                print(model_name + " input_chunk_length/output_chunk_length optimization:")
                for result in results:
                    print(*result)
            model_args['input_chunk_length'] = results[0][1]
            model_args['output_chunk_length'] = results[0][2]
            metric_result = results[0][0]
    else:
        # Not TCNModel
        if verbose: print("#+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-")
        if verbose: print("Coarse optimization (not TCNModel)..")
        i = powers_of_two_in_range(2, input_chunk_length_max)
        i.append(input_chunk_length_max)
        if output_chunk_length_max < 2:
            o = [1]
        else:
            o = powers_of_two_in_range(2, output_chunk_length_max)
            o.append(output_chunk_length_max)
            i.sort(reverse=True)
            o.sort(reverse=True)
            if len(i) == 0: raise Exception("Unable to find input_chunk_length between 1 and " + str(input_chunk_length_max) + "(input_chunk_length_max)")
            if len(o) == 0: raise Exception("Unable to find output_chunk_length between 1 and " + str(output_chunk_length_max) + "(output_chunk_length_max)")
        if verbose: print("input_chunk_length:", i)
        if verbose: print("output_chunk_length:", o)
        parameters = {'input_chunk_length': i, 'output_chunk_length': o}
        results = []
        model_args = {}
        for input_chunk_length in i:
            for output_chunk_length in o:
                if (input_chunk_length+output_chunk_length) <= sum_chunk_length_limit:
                    print("get_darts_tfm_arguments_optimized() calling darts_train_fit_pred_tfm()")
                    metric_result, pred, model = darts_train_fit_pred_tfm(ts, train, val_series, test, pred_input, pred_steps, 
                                                                        model_class, min_max_scale, metric, 
                                                                        input_chunk_length=input_chunk_length, output_chunk_length=output_chunk_length, 
                                                                        past_covariates=past_covariates, val_past_covariates=val_past_covariates, future_covariates=future_covariates, val_future_covariates=val_future_covariates, 
                                                                        verbose=verbose)
                    results.append([metric_result, input_chunk_length, output_chunk_length])
                    if verbose: print(model_name, "input_chunk_length: ", input_chunk_length, "\toutput_chunk_length: ", output_chunk_length, "\tmetric: ", metric_result)
        if len(results) == 0: raise Exception("ERROR: No optimization results could be obtained for " + model_name + " using get_darts_tfm_arguments_optimized()")
        # Sort the results and get the one with the lowest value (best) metric
        results = sorted(results, key=itemgetter(0))
        if devmt:
            print(model_name + " input_chunk_length/output_chunk_length optimization:")
            for result in results:
                print(*result)
        model_args['input_chunk_length'] = results[0][1]
        model_args['output_chunk_length'] = results[0][2]
        if verbose: print("Coarse optimization results:")
        if verbose: print("input_chunk_length:", model_args['input_chunk_length'])
        if verbose: print("output_chunk_length:", model_args['output_chunk_length'])
        metric_result = results[0][0]

        if coarse_only == False:
            # Fine optimization of model arguments
            if verbose: print("#+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-")
            if verbose: print("Fine optimization..")
            if output_chunk_length_max < 2:
                i = list(range(model_args['input_chunk_length']-2**2, model_args['input_chunk_length']+2**2+1))
                # Filter based on input_chunk_length_max
                i = list(filter(lambda x: x<=input_chunk_length_max, i))
                i = list(filter(lambda x: x>0, i))
                if len(i) == 0: i = list(range(model_args['input_chunk_length']-2**2, model_args['input_chunk_length']+1))
                if len(i) == 0: i = [model_args['input_chunk_length']]
                i = list(filter(lambda x: x<=input_chunk_length_max, i))
                i = list(filter(lambda x: x>0, i))
                o = [1]
            else:
                i = list(range(model_args['input_chunk_length']-2**2, model_args['input_chunk_length']+2**2+1))
                o = list(range(model_args['output_chunk_length']-2**2, model_args['output_chunk_length']+2**2+1))
                # Filter based on input_chunk_length_max and output_chunk_length_max
                i = list(filter(lambda x: x<=input_chunk_length_max, i))
                i = list(filter(lambda x: x>0, i))
                o = list(filter(lambda x: x<=output_chunk_length_max, o))
                o = list(filter(lambda x: x>0, o))
                if len(i) == 0: i = list(range(model_args['input_chunk_length']-2**2, model_args['input_chunk_length']+1))
                if len(o) == 0: o = list(range(model_args['output_chunk_length']-2**2, model_args['output_chunk_length']+1))
                if len(i) == 0: i = [model_args['input_chunk_length']]
                if len(o) == 0: o = [model_args['output_chunk_length']]
                i = list(filter(lambda x: x<=input_chunk_length_max, i))
                i = list(filter(lambda x: x>0, i))
                o = list(filter(lambda x: x<=output_chunk_length_max, o))
                o = list(filter(lambda x: x>0, o))
            if len(i) == 0: i = [model_args['input_chunk_length']]
            if len(o) == 0: o = [model_args['output_chunk_length']]
            i.sort(reverse=True)
            o.sort(reverse=True)
            if verbose: print("input_chunk_length:", i)
            if verbose: print("output_chunk_length:", o)
            parameters = {'input_chunk_length': i, 'output_chunk_length': o}
            results = []
            model_args = {}
            for input_chunk_length in i:
                for output_chunk_length in o:
                    if (input_chunk_length+output_chunk_length) <= sum_chunk_length_limit:
                        metric_result, pred, model = darts_train_fit_pred_tfm(ts, train, val_series, test, pred_input, pred_steps, 
                                                                            model_class, min_max_scale, metric,
                                                                            input_chunk_length=input_chunk_length, output_chunk_length=output_chunk_length, 
                                                                            past_covariates=past_covariates, val_past_covariates=val_past_covariates, future_covariates=future_covariates, val_future_covariates=val_future_covariates,
                                                                            verbose=verbose)
                        results.append([metric_result, input_chunk_length, output_chunk_length])
                        if verbose: print(model_name, "input_chunk_length: ", input_chunk_length, "\toutput_chunk_length: ", output_chunk_length, "\tmetric: ", metric_result)
            if len(results) == 0: raise Exception("ERROR: No optimization results could be obtained for " + model_name + " using get_darts_tfm_arguments_optimized()")
            # Sort the results and get the one with the lowest value (best) metric
            results = sorted(results, key=itemgetter(0))
            if devmt:
                print(model_name + " input_chunk_length/output_chunk_length optimization:")
                for result in results:
                    print(*result)
            model_args['input_chunk_length'] = results[0][1]
            model_args['output_chunk_length'] = results[0][2]
            metric_result = results[0][0]


    if verbose: print("The optimized input_chunk_length augument is ", model_args['input_chunk_length'], " based on minimizing the RMSE")
    if verbose: print("The optimized output_chunk_length augument is ", model_args['output_chunk_length'], " based on minimizing the RMSE")
    if verbose: print("metric_result: ", metric_result)

    t_elapsed = round(perf_counter()-t_start,3) 
    if verbose: print(str(t_elapsed) + " sec to coarse & fine optimize model parameters")   

    return model_args['input_chunk_length'], model_args['output_chunk_length'], metric_result, t_elapsed


def darts_tfm_historical_forecasts(model_name="", model_class=None, input_chunk_length=None, output_chunk_length=None, sum_chunk_length_limit=None, ts=None, past_covariates=None, future_covariates=None, forecast_horizon=None, stride=None, metric=None, min_max_scale=True, retrain=True, verbose=True):
    # Train, fit, and predict a Local Forecasting Model (LFM) using backtesting (.historical_forecasts()).
    # If min_max_scale == True, scales the data between 0.0 and +1.0
    # Returns: model, model_class, input_chunk_length, output_chunk_length, train_length, start, forecast_horizon, stride, enable_optimization, series, pred, scaler, metric_result, t_elapsed

    from time import perf_counter
    from datetime import timedelta
    from operator import itemgetter
    import pandas as pd
    import numpy as np

    from darts import TimeSeries, concatenate
    from darts.dataprocessing.transformers.scaler import Scaler
    scaler = Scaler()  # default uses sklearn's MinMaxScaler.  Does NOT support multivariate series

    # Valiate the inputs
    if not isinstance(model_name, str): raise Exception("Value Error: model_name must be string type corresponding to the Darts model name")
    if model_class is None: raise Exception("Value Error: you must pass a Darts model for the argument model_class")
    if metric is None: raise Exception("Value Error: argument metric must not be None.  It should be a Darts metric.  See: from darts.metrics import [metric]")
    if not isinstance(min_max_scale, bool): raise Exception("Value Error: min_max_scale must be boolean type with value of True or False")
    if not isinstance(verbose, bool): raise Exception("Value Error: verbose must be boolean type with value of True or False")
    if not isinstance(input_chunk_length, int): raise Exception("Value Error: input_chunk_length must be int type >= 1")
    if not isinstance(output_chunk_length, int): raise Exception("Value Error: output_chunk_length must be int type >= 1", type(output_chunk_length), output_chunk_length)
    if not isinstance(sum_chunk_length_limit, int): raise Exception("Value Error: sum_chunk_length_limit must be int type >= 1")

    # Validate any series passed
    if ts is None or not isinstance(ts, TimeSeries): raise Exception("Value Error: ts is not a Darts TimeSeries")
    #if pred_steps is None or not isinstance(pred_steps, int): raise Exception("Value Error: pred_steps is not None or an integer")

    # past_covariates, future_covariates
    if not past_covariates is None and not isinstance(past_covariates, TimeSeries): raise Exception("Value Error: past_covariates is not a Darts TimeSeries")
    if not future_covariates is None and not isinstance(future_covariates, TimeSeries): raise Exception("Value Error: future_covariates is not a Darts TimeSeries")

    # IMPORTANT:  forecast_horizon (n) > output_chunk_length + 1
    if forecast_horizon < output_chunk_length: print("WARNING: forecast_horizon < output_chunk_length will cause autoregression to be employed")
    if not isinstance(forecast_horizon, int): raise Exception("Value Error: forecast_horizon must be int type >= 1")
    if not isinstance(stride, int): raise Exception("Value Error: stride must be int type >= 1")


    #if not model_save_path is None and not model_save_path.exists(): raise Exception("Folder specified by 'path_file' doesn't exist: "+ str(model_save_path))

    t_start = perf_counter()

    if model_name == "TCNModel":

        if not future_covariates is None:
            if verbose: print("WARNING: future covariates specified, but they are not supported by TCNModel.  Ignoring future covariates.")
            future_covariates = None

        model = model_class(input_chunk_length=input_chunk_length, output_chunk_length=output_chunk_length)

        # Trim the target series (for training & prediction) to the maximum length (limited by past/future coveriates & input_chunk_length/output_chunk_length)
        if past_covariates is None and future_covariates is None:
            series = ts
        else:
            # Series ts must be smaller than future_covariates & past covariates.  
            #print("Min series length for past & future covariates: ", len(ts)-input_chunk_length-output_chunk_length+1)
            series = ts[input_chunk_length+1:len(ts)-output_chunk_length]

        if min_max_scale == True: 
            series = scaler.fit_transform(series)
            if not past_covariates is None: past_covariates = scaler.transform(past_covariates)
            if not future_covariates is None: future_covariates = scaler.transform(future_covariates)
        #if isinstance(series, TimeSeries): print("\nseries: ", series.time_index.min(), " ..", series.time_index.max(), "\tlength: ", len(series.time_index))  

        # Define the minimum start position (defines the length of the training series).
        start = None
        train_length = None
        # `start` must be either `float`, `int`, `pd.Timestamp` or `None`
        if isinstance(ts.time_index, pd.DatetimeIndex):
            #print("series.time_index.freqstr:",series.time_index.freqstr)       # D
            start = series.time_index.min() + timedelta(days=int(input_chunk_length + output_chunk_length + 1))
        else:
            start = series.time_index.min() + input_chunk_length + output_chunk_length + 1
        #if verbose: print("start:", start)

        #t_start = perf_counter()
        # Per Dennis Bader, use retrain=False, BUT this requires .fit() prior to calling .historical_forecasts()
        pred = model.historical_forecasts(series=series, past_covariates=past_covariates, future_covariates=future_covariates, num_samples=1, train_length=train_length, start=start, start_format='value', forecast_horizon=forecast_horizon, stride=stride, retrain=retrain, overlap_end=False, last_points_only=False, verbose=False, enable_optimization=model.supports_optimized_historical_forecasts)
        #t_elapsed = round(perf_counter()-t_start,3) 
        #if verbose: print(str(t_elapsed) + " sec to execute .historical_forecasts() for model " + model_name) 

        # If last_points_only=False .historical_forecasts() returns a list of forecasts.  Use concatenate() to join them together.
        # Keep all predicted values from each forecast by setting last_points_only=False
        # WARNING: If stride != forecast_horizon then an error is thrown when concatenate() is anticipated
        #from darts import TimeSeries, concatenate
        if isinstance(pred, list): 
            pred = concatenate(pred)

        #if isinstance(pred, TimeSeries) and verbose==True: print("\npred: ", pred.time_index.min(), " ..", pred.time_index.max(), "\tlength: ", len(pred.time_index))  

        # Calculate the root mean square error (RMSE)
        ground_truth = ts.slice_intersect(pred)
        if min_max_scale:
            # Scale pred back to original scale
            pred = scaler.inverse_transform(pred)
        metric_result = metric(ground_truth, pred)

        if verbose: print(model_name, "input_chunk_length:", input_chunk_length, "\tstride = forecast_horizon:", forecast_horizon, "\toutput_chunk_length:", output_chunk_length, "\tmetric:", round(metric_result,5), "\tstart:", start, "\t", series.time_index.min(), " ..", series.time_index.max(), "\tlength: ", len(series.time_index), "\n")

    else:
        # Not TCNModel

        model = model_class(input_chunk_length=input_chunk_length, output_chunk_length=output_chunk_length)

        # Trim the target series (for training & prediction) to the maximum length (limited by past/future coveriates & input_chunk_length/output_chunk_length)
        if past_covariates is None and future_covariates is None:
            series = ts
        else:
            # Series ts must be smaller than future_covariates & past covariates.  
            #print("Min series length for past & future covariates: ", len(ts)-input_chunk_length-output_chunk_length+1)
            series = ts[input_chunk_length+1:len(ts)-output_chunk_length]

        if min_max_scale == True: 
            series = scaler.fit_transform(series)
            if not past_covariates is None: past_covariates = scaler.transform(past_covariates)
            if not future_covariates is None: future_covariates = scaler.transform(future_covariates)
            #if isinstance(series, TimeSeries): print("\nseries: ", series.time_index.min(), " ..", series.time_index.max(), "\tlength: ", len(series.time_index))  

        # Define the minimum start position (defines the length of the training series).
        start = None
        train_length = None
        # `start` must be either `float`, `int`, `pd.Timestamp` or `None`
        if isinstance(ts.time_index, pd.DatetimeIndex):
            #print("series.time_index.freqstr:",series.time_index.freqstr)       # D
            start = series.time_index.min() + timedelta(days=int(input_chunk_length + output_chunk_length + 1))
        else:
            start = series.time_index.min() + input_chunk_length + output_chunk_length + 1
        #if verbose: print("start:", start)

        #t_start = perf_counter()
        # Per Dennis Bader, use retrain=False.  
        pred = model.historical_forecasts(series=series, past_covariates=past_covariates, future_covariates=future_covariates, num_samples=1, train_length=train_length, start=start, start_format='value', forecast_horizon=forecast_horizon, stride=stride, retrain=retrain, overlap_end=False, last_points_only=False, verbose=False, enable_optimization=model.supports_optimized_historical_forecasts)
        #t_elapsed = round(perf_counter()-t_start,3) 
        #if verbose: print(str(t_elapsed) + " sec to execute .historical_forecasts() for model " + model_name) 

        # If last_points_only=False .historical_forecasts() returns a list of forecasts.  Use concatenate() to join them together.
        # Keep all predicted values from each forecast by setting last_points_only=False
        # WARNING: If stride != forecast_horizon then an error is thrown when concatenate() is anticipated
        #from darts import TimeSeries, concatenate
        if isinstance(pred, list): 
            pred = concatenate(pred)
        
        #if isinstance(pred, TimeSeries) and verbose==True: print("\npred: ", pred.time_index.min(), " ..", pred.time_index.max(), "\tlength: ", len(pred.time_index))  

        # Calculate the root mean square error (RMSE)
        ground_truth = ts.slice_intersect(pred)
        if min_max_scale:
            # Scale pred back to original scale
            pred = scaler.inverse_transform(pred)
        metric_result = metric(ground_truth, pred)
        
        if verbose: print(model_name, "input_chunk_length:", input_chunk_length, "\tstride = forecast_horizon:", forecast_horizon, "\toutput_chunk_length:", output_chunk_length, "\tmetric:", round(metric_result,5), "\tstart:", start, "\t", series.time_index.min(), " ..", series.time_index.max(), "\tlength: ", len(series.time_index), "\n")
        if verbose: print(model_name, "input_chunk_length: ", input_chunk_length, "\toutput_chunk_length: ", output_chunk_length, "\tmetric: ", metric_result)

    t_elapsed = round(perf_counter()-t_start,3) 
    if verbose: print(str(t_elapsed) + " sec to train,fit,predict TFM")   

    enable_optimization=model.supports_optimized_historical_forecasts
    return model, train_length, start, enable_optimization, series, pred, scaler, metric_result, t_elapsed


def get_darts_tfm_arguments_optimized_via_backtesting(model_name="", model_class=None, ts=None, past_covariates=None, future_covariates=None, metric=None, min_max_scale=True, coarse_only=True, model_save_path=None, verbose=True):
    # Optimizes input_chunk_length and output_chunk_length within the limits
    # of input_chunk_length_max, output_chunk_length, and sum_chunk_length_limit.
    # For the coarse optimization, a range of 1 to the maximum size in steps of 
    # powers of 2 (1,2,4,8,16,..) is employed.
    # If coarse_only=False, then a fine optimization continues and optimizes
    # between the powers of two found on either side of the coarse optimization,
    # but within the limits of input_chunk_length_max, output_chunk_length, and sum_chunk_length_limit.
    # Any future covariates specified for the TCNModel are ignored.

    # RETURNS: input_chunk_length, output_chunk_length, metric_result, t_elapsed, model

    # Use the function get_darts_tfm_arguments() to get the following required inputs:
    #   input_chunk_length_max
    #   output_chunk_length_max
    #   sum_chunk_length_limit

    # Does not use .gridsearch() because of the relational limitations between
    # input_chunk_length & output_chunk_length will raise errors with several models.

    from time import perf_counter
    from datetime import timedelta
    from operator import itemgetter
    import pandas as pd
    import numpy as np

    from darts import TimeSeries, concatenate
    from darts.dataprocessing.transformers.scaler import Scaler
    scaler = Scaler()  # default uses sklearn's MinMaxScaler.  Does NOT support multivariate series

    # Valiate the inputs
    if not isinstance(model_name, str): raise Exception("Value Error: model_name must be string type corresponding to the Darts model name")
    if model_class is None: raise Exception("Value Error: you must pass a Darts model for the argument model_class")
    if metric is None: raise Exception("Value Error: argument metric must not be None.  It should be a Darts metric.  See: from darts.metrics import [metric]")
    if not isinstance(min_max_scale, bool): raise Exception("Value Error: min_max_scale must be boolean type with value of True or False")
    if not isinstance(verbose, bool): raise Exception("Value Error: verbose must be boolean type with value of True or False")

    # Validate any series passed
    if ts is None or not isinstance(ts, TimeSeries): raise Exception("Value Error: ts is not a Darts TimeSeries")
    # past_covariates, future_covariates
    if not past_covariates is None and not isinstance(past_covariates, TimeSeries): raise Exception("Value Error: past_covariates is not a Darts TimeSeries")
    if not future_covariates is None and not isinstance(future_covariates, TimeSeries): raise Exception("Value Error: future_covariates is not a Darts TimeSeries")

    if not model_save_path is None and not model_save_path.exists(): raise Exception("Folder specified by 'path_file' doesn't exist: "+ str(model_save_path))

    #seasonal_periods = ts_seasonal_periods(ts=ts, sort_ascending=True, unique=True)
    #print("*********************************")
    #print("seasonal_periods = []")
    #seasonal_periods = []
    #print("*********************************")

    devmt = False        # shortens execution (for logig testing only)

    if verbose: print("\n---------------------------------------------------------------------------")
    if verbose: print("get_darts_tfm_arguments_optimized_via_backtesting()")
    t_start = perf_counter()

    sum_chunk_length_limit = len(ts)
    if verbose: print("sum_chunk_length_limit:", sum_chunk_length_limit)

    # Establish the forecast_horizon, stride, input_chunk_length, output_chunk_length
    # For a seasonal_period:
    #   forecast_horizon is specified by user (for the task at hand), BUT should be >= seasonal_period
    #   stride = forecast_horizon
    #   output_chunk_length >= forecast_horizon
    #   input_chunk_length = multiple of any seasonal period, or = forecast_horizon
    seasonal_periods = ts_seasonal_periods(ts=ts, sort_ascending=False, unique=False)
    #seasonal_periods = []       # for testing only
    if verbose: print("seasonal_periods:",seasonal_periods)
    if len(seasonal_periods) == 0:
        # No seasonality found
        divisors_of_ts = divisors_of_int(n=len(ts), smallest=True)
        #print("divisors_of_ts", divisors_of_ts)
        # Get a divisor of the series ts from the middle of the list in divisors_of_ts
        forecast_horizon = len(ts)//divisors_of_ts[len(divisors_of_ts)//2]
        input_chunk_length_max = forecast_horizon
    else:
        # Seasonality found
        # Each forecast horizon chunk should be at least the length of the target series maximum seasonal period.
        # If less than the seasonal peroid, it takes excessively long to solve (but the result is decent).
        forecast_horizon = max(seasonal_periods)*1      
        # Define the input_chunk_length from a multiple of the seasonal periods
        input_chunk_length_max = []
        for sp in seasonal_periods:
            l = list(range(sp, forecast_horizon+1, sp))
            input_chunk_length_max.extend(l)
        # Remove any duplicates and sort
        input_chunk_length_max = list(set(input_chunk_length_max)) 
        input_chunk_length_max.sort(reverse=False)   
        #print("input_chunk_length_max:", input_chunk_length_max)
        input_chunk_length_max = max(seasonal_periods)


    if model_name == "TCNModel":
        # Cannot use .gridsearch() for TCNModel because: 
        #   The output length must be strictly smaller than the input length
        #   The kernel size (size of every kernel in a convolutional layer) must be strictly smaller than the input length.  kererl_size = 3 by default 
        #   output_chunk_length < kernel_size < input_chunk_length
        if not future_covariates is None:
            if verbose: print("WARNING: future covariates specified, but they are not supported by TCNModel.  Ignoring future covariates.")
            future_covariates = None
        if verbose: print("#+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-")
        if verbose: print("Coarse optimization of " + model_name + "..")
        kernel_size = 3     # default for TCNModel

        output_chunk_length_max = forecast_horizon      # output_chunk_length >= forecast_horizon to avoid autoregression
        if verbose: print("input_chunk_length_max:", input_chunk_length_max)
        if verbose: print("output_chunk_length_max:", output_chunk_length_max)
        stride = forecast_horizon
        if verbose: print("stride = forecast_horizon = ", forecast_horizon)

        if len(seasonal_periods) == 0:
            # No seasonal periods found in series ts
            i = powers_of_two_in_range(2, input_chunk_length_max)
            i.append(input_chunk_length_max)
        else:
            # seasonal periods found in ts
            i = []
            for sp in seasonal_periods:
                l = list(range(sp, input_chunk_length_max+1, sp))
                i.extend(l)
            # Remove any duplicates
            i = list(set(i))    
        # Filter by kernel_size
        i = list(filter(lambda x: kernel_size<x, i))

        if output_chunk_length_max < 2:
            o = [1]
        else:
            o = powers_of_two_in_range(2, output_chunk_length_max)
            o.append(output_chunk_length_max)
        o.append(output_chunk_length_max)
        # Remove any duplicates
        o = list(set(o))    
       
        i.sort(reverse=True)
        o.sort(reverse=True)
        if len(i) == 0: raise Exception("Unable to find input_chunk_length between " + str(kernel_size) + " (kernel_size) and " + str(input_chunk_length_max) + "(input_chunk_length_max)")
        if verbose: print("input_chunk_length:", i)
        if verbose: print("output_chunk_length:", o)

        results = []
        model_args = {}
        for input_chunk_length in i:
            for output_chunk_length in o:
                if output_chunk_length < input_chunk_length and (input_chunk_length+output_chunk_length) <= sum_chunk_length_limit:

                    model, train_length, start, enable_optimization, series, pred, scaler, metric_result, t_elapsed = darts_tfm_historical_forecasts(model_name=model_name, model_class=model_class, input_chunk_length=input_chunk_length, output_chunk_length=output_chunk_length, sum_chunk_length_limit=len(ts), ts=ts, past_covariates=past_covariates, future_covariates=future_covariates, forecast_horizon=forecast_horizon, stride=stride, metric=metric, min_max_scale=min_max_scale, retrain=True, verbose=True)

                    if isinstance(pred, TimeSeries) and verbose==True: print("\npred: ", pred.time_index.min(), " ..", pred.time_index.max(), "\tlength: ", len(pred.time_index))  

                    results.append([metric_result, input_chunk_length, output_chunk_length, model])
                    if verbose: print(model_name, "input_chunk_length:", input_chunk_length, "\tstride = forecast_horizon:", forecast_horizon, "\toutput_chunk_length:", output_chunk_length, "\tmetric:", round(metric_result,5), "\tstart:", start, "\t", series.time_index.min(), " ..", series.time_index.max(), "\tlength: ", len(series.time_index), "\n")
        
        if len(results) == 0: raise Exception("ERROR: No optimization results could be obtained for " + model_name + " using get_darts_tfm_arguments_optimized()")
        # Sort the results and get the one with the lowest value (best) metric
        results = sorted(results, key=itemgetter(0))
        if devmt:
            print(model_name + " input_chunk_length/output_chunk_length optimization:")
            for result in results:
                print(*result)
        model_args['input_chunk_length'] = results[0][1]
        model_args['output_chunk_length'] = results[0][2]
        if verbose: print("Coarse optimization results:")
        if verbose: print("input_chunk_length:", model_args['input_chunk_length'])
        if verbose: print("output_chunk_length:", model_args['output_chunk_length'])
        metric_result = results[0][0]
        model = results[0][3]

        if coarse_only == False:
            # Fine optimization of model arguments
            if verbose: print("#+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-")
            if verbose: print("Fine optimization of " + model_name + "..")
            i = list(range(model_args['input_chunk_length']-2**2, model_args['input_chunk_length']+2**2+1))
            o = list(range(model_args['output_chunk_length']-2**2, model_args['output_chunk_length']+2**2+1))
            # Filter based on input_chunk_length_max and output_chunk_length_max
            i = list(filter(lambda x: x<=input_chunk_length_max, i))
            i = list(filter(lambda x: x>0, i))
            o = list(filter(lambda x: x<=output_chunk_length_max, o))
            o = list(filter(lambda x: x>0, o))
            # Filter by kernel_size
            i = list(filter(lambda x: kernel_size<x, i))
            if len(i) == 0: i = list(range(model_args['input_chunk_length']-2**2, model_args['input_chunk_length']+1))
            if len(o) == 0: o = list(range(model_args['output_chunk_length']-2**2, model_args['output_chunk_length']+1))
            if len(i) == 0: i = [model_args['input_chunk_length']]
            if len(o) == 0: o = [model_args['output_chunk_length']]
            i = list(filter(lambda x: x<=input_chunk_length_max, i))
            i = list(filter(lambda x: x>0, i))
            o = list(filter(lambda x: x<=output_chunk_length_max, o))
            o = list(filter(lambda x: x>0, o))
            # Filter by kernel_size
            i = list(filter(lambda x: kernel_size<x, i))
            if len(i) == 0: i = [model_args['input_chunk_length']]
            if len(o) == 0: o = [model_args['output_chunk_length']]
            i.sort(reverse=True)
            o.sort(reverse=True)
            if verbose: print("input_chunk_length:", i)
            if verbose: print("output_chunk_length:", o)
            parameters = {'input_chunk_length': i, 'output_chunk_length': o}
            results = []
            model_args = {}
            for input_chunk_length in i:
                for output_chunk_length in o:
                    if output_chunk_length < input_chunk_length and (input_chunk_length+output_chunk_length) <= sum_chunk_length_limit:
                        # IMPORTANT:  forecast_horizon (n) > output_chunk_length + 1
                        forecast_horizon = output_chunk_length + 1              # n>output_chunk_length     lower = better fit
                        stride = forecast_horizon

                        model, train_length, start, enable_optimization, series, pred, scaler, metric_result, t_elapsed = darts_tfm_historical_forecasts(model_name=model_name, model_class=model_class, input_chunk_length=input_chunk_length, output_chunk_length=output_chunk_length, sum_chunk_length_limit=len(ts), ts=ts, past_covariates=past_covariates, future_covariates=future_covariates, forecast_horizon=forecast_horizon, stride=stride, metric=metric, min_max_scale=min_max_scale, retrain=True, verbose=True)

                        results.append([metric_result, input_chunk_length, output_chunk_length, model])
                        if verbose: print(model_name, "input_chunk_length:", input_chunk_length, "\tstride = forecast_horizon:", forecast_horizon, "\toutput_chunk_length:", output_chunk_length, "\tmetric:", round(metric_result,5), "\tstart:", start, "\t", series.time_index.min(), " ..", series.time_index.max(), "\tlength: ", len(series.time_index), "\n")
                        #logging.info(model_name + "input_chunk_length:" + str(input_chunk_length) + "\tstride = forecast_horizon:" + str(forecast_horizon) + "\toutput_chunk_length:" + str(output_chunk_length) + "\tmetric:" + str(round(metric_result,5)))
            if len(results) == 0: raise Exception("ERROR: No optimization results could be obtained for " + model_name + " using get_darts_tfm_arguments_optimized()")
            # Sort the results and get the one with the lowest value (best) metric
            results = sorted(results, key=itemgetter(0))
            if devmt:
                print(model_name + " input_chunk_length/output_chunk_length optimization:")
                for result in results:
                    print(*result)
            model_args['input_chunk_length'] = results[0][1]
            model_args['output_chunk_length'] = results[0][2]
            metric_result = results[0][0]
            model = results[0][3]
    else:
        # Not TCNModel
        if verbose: print("#+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-")
        if verbose: print("Optimization of " + model_name + "..")

        output_chunk_length_max = forecast_horizon      # output_chunk_length >= forecast_horizon to avoid autoregression
        if verbose: print("input_chunk_length_max:", input_chunk_length_max)
        if verbose: print("output_chunk_length_max:", output_chunk_length_max)
        stride = forecast_horizon
        if verbose: print("stride = forecast_horizon = ", forecast_horizon)

        if len(seasonal_periods) == 0:
            # No seasonal periods found in series ts
            i = powers_of_two_in_range(2, input_chunk_length_max)
            i.append(input_chunk_length_max)
        else:
            # seasonal periods found in ts
            i = []
            for sp in seasonal_periods:
                l = list(range(sp, input_chunk_length_max+1, sp))
                i.extend(l)
            # Remove any duplicates
            i = list(set(i))    
        if output_chunk_length_max < 2:
            o = [1]
        else:
            o = powers_of_two_in_range(2, output_chunk_length_max)
            o.append(output_chunk_length_max)
        # Remove any duplicates
        o = list(set(o))    
        #i = list(range(1, input_chunk_length_max+2**2+1))
        #o = list(range(1, output_chunk_length_max+2**2+1))
        i.sort(reverse=True)
        o.sort(reverse=True)
        if len(i) == 0: raise Exception("Unable to find input_chunk_length between 1 and " + str(input_chunk_length_max) + "(input_chunk_length_max)")
        if len(o) == 0: raise Exception("Unable to find output_chunk_length between 1 and " + str(output_chunk_length_max) + "(output_chunk_length_max)")
            
        if verbose: print("input_chunk_length:", i)
        if verbose: print("output_chunk_length:", o)
        results = []
        model_args = {}
        for input_chunk_length in i:
            for output_chunk_length in o:
                if (input_chunk_length+output_chunk_length) <= sum_chunk_length_limit:

                    model, train_length, start, enable_optimization, series, pred, scaler, metric_result, t_elapsed = darts_tfm_historical_forecasts(model_name=model_name, model_class=model_class, input_chunk_length=input_chunk_length, output_chunk_length=output_chunk_length, sum_chunk_length_limit=len(ts), ts=ts, past_covariates=past_covariates, future_covariates=future_covariates, forecast_horizon=forecast_horizon, stride=stride, metric=metric, min_max_scale=min_max_scale, retrain=True, verbose=True)

                    results.append([metric_result, input_chunk_length, output_chunk_length, model])
                    if verbose: print(model_name, "input_chunk_length:", input_chunk_length, "\tstride = forecast_horizon:", forecast_horizon, "\toutput_chunk_length:", output_chunk_length, "\tmetric:", round(metric_result,5), "\tstart:", start, "\t", series.time_index.min(), " ..", series.time_index.max(), "\tlength: ", len(series.time_index), "\n")
                    #logging.info(model_name + "input_chunk_length:" + str(input_chunk_length) + "\tstride = forecast_horizon:" + str(forecast_horizon) + "\toutput_chunk_length:" + str(output_chunk_length) + "\tmetric:" + str(round(metric_result,5)))

        if verbose: print("input_chunk_length:", i)
        if verbose: print("output_chunk_length:", o)
        if len(results) == 0: raise Exception("ERROR: No optimization results could be obtained for " + model_name + " using get_darts_tfm_arguments_optimized()")
        # Sort the results and get the one with the lowest value (best) metric
        results = sorted(results, key=itemgetter(0))
        print("\n", model_name + " input_chunk_length/output_chunk_length optimization:")
        for result in results:
            print(*result)
        model_args['input_chunk_length'] = results[0][1]
        model_args['output_chunk_length'] = results[0][2]
        metric_result = results[0][0]
        model = results[0][3]

        if len(seasonal_periods) == 0:
        #if coarse_only == False:
            # Fine optimization of model arguments
            if verbose: print("#+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-")
            if verbose: print("Fine optimization..")
            if output_chunk_length_max < 2:
                i = list(range(model_args['input_chunk_length']-2**2, model_args['input_chunk_length']+2**2))
                # Filter based on input_chunk_length_max
                i = list(filter(lambda x: x<=input_chunk_length_max, i))
                i = list(filter(lambda x: x>0, i))
                if len(i) == 0: i = list(range(model_args['input_chunk_length']-2**2, model_args['input_chunk_length']+1))
                if len(i) == 0: i = [model_args['input_chunk_length']]
                i = list(filter(lambda x: x<=input_chunk_length_max, i))
                i = list(filter(lambda x: x>0, i))
                o = [1]
            else:
                #i = list(range(model_args['input_chunk_length']-2**2, model_args['input_chunk_length']+2**2+1))
                #o = list(range(model_args['output_chunk_length']-2**2, model_args['output_chunk_length']+2**2+1))
                i = list(range(model_args['input_chunk_length']-2**2, model_args['input_chunk_length']+2**2))
                o = list(range(model_args['output_chunk_length']-2**2, model_args['output_chunk_length']+2**2))
                # Filter based on input_chunk_length_max and output_chunk_length_max
                i = list(filter(lambda x: x<=input_chunk_length_max, i))
                i = list(filter(lambda x: x>0, i))
                o = list(filter(lambda x: x<=output_chunk_length_max, o))
                o = list(filter(lambda x: x>0, o))
                if len(i) == 0: i = list(range(model_args['input_chunk_length']-2**2, model_args['input_chunk_length']+1))
                if len(o) == 0: o = list(range(model_args['output_chunk_length']-2**2, model_args['output_chunk_length']+1))
                if len(i) == 0: i = [model_args['input_chunk_length']]
                if len(o) == 0: o = [model_args['output_chunk_length']]
                i = list(filter(lambda x: x<=input_chunk_length_max, i))
                i = list(filter(lambda x: x>0, i))
                o = list(filter(lambda x: x<=output_chunk_length_max, o))
                o = list(filter(lambda x: x>0, o))
            if len(i) == 0: i = [model_args['input_chunk_length']]
            if len(o) == 0: o = [model_args['output_chunk_length']]
            i.sort(reverse=True)
            o.sort(reverse=True)
            if verbose: print("input_chunk_length:", i)
            if verbose: print("output_chunk_length:", o)
            parameters = {'input_chunk_length': i, 'output_chunk_length': o}
            results = []
            model_args = {}
            for input_chunk_length in i:
                for output_chunk_length in o:
                    if (input_chunk_length+output_chunk_length) <= sum_chunk_length_limit:

                        model, train_length, start, enable_optimization, series, pred, scaler, metric_result, t_elapsed = darts_tfm_historical_forecasts(model_name=model_name, model_class=model_class, input_chunk_length=input_chunk_length, output_chunk_length=output_chunk_length, sum_chunk_length_limit=len(ts), ts=ts, past_covariates=past_covariates, future_covariates=future_covariates, forecast_horizon=forecast_horizon, stride=stride, metric=metric, min_max_scale=min_max_scale, retrain=True, verbose=True)
                        
                        results.append([metric_result, input_chunk_length, output_chunk_length, model])
                        if verbose: print(model_name, "input_chunk_length:", input_chunk_length, "\tstride = forecast_horizon:", forecast_horizon, "\toutput_chunk_length:", output_chunk_length, "\tmetric:", round(metric_result,5), "\tstart:", start, "\t", series.time_index.min(), " ..", series.time_index.max(), "\tlength: ", len(series.time_index), "\n")
                        if verbose: print(model_name, "input_chunk_length: ", input_chunk_length, "\toutput_chunk_length: ", output_chunk_length, "\tmetric: ", metric_result)

            if len(results) == 0: raise Exception("ERROR: No optimization results could be obtained for " + model_name + " using get_darts_tfm_arguments_optimized()")
            # Sort the results and get the one with the lowest value (best) metric
            results = sorted(results, key=itemgetter(0))
            print("\n", model_name + " input_chunk_length/output_chunk_length optimization:")
            for result in results:
                print(*result)
            model_args['input_chunk_length'] = results[0][1]
            model_args['output_chunk_length'] = results[0][2]
            metric_result = results[0][0]
            model = results[0][3]

    if verbose: print("The optimized input_chunk_length augument is ", model_args['input_chunk_length'], " based on minimizing the RMSE")
    if verbose: print("The optimized output_chunk_length augument is ", model_args['output_chunk_length'], " based on minimizing the RMSE")
    if verbose: print("metric_result: ", metric_result)

    t_elapsed = round(perf_counter()-t_start,3) 
    if verbose: print(str(t_elapsed) + " sec to coarse & fine optimize model parameters")   

    return model_args['input_chunk_length'], model_args['output_chunk_length'], metric_result, t_elapsed, model


def slice_tfm_covariates_for_model_training(ts=None, train=None, val_series=None, test=None, pred_input=None, pred_steps=None, min_max_scale=True, output_chunk_length=None, past_covariates=None, val_past_covariates=None, future_covariates=None, val_future_covariates=None, verbose=True):
    # Returns Darts covariates derived from the series passed as past_covariates, val_past_covariates, future_covariates, val_future_covariates
    # sliced to the optimal length for Torch Forecasting Model .fit() and .predict() methods. 
    # NOTE:  If you need sliced covariates for a trained model, use slice_tfm_covariates_for_trained_model() instead.
    # Set any of the covariate arguments to None if not desired. 
    # The series ts, val_series, test, and pred_input and the value pred_steps are used for calculation only, and val_series pred_input may be None.
    # input_chunk_length and output_chunk_length must be assigned.  Use get_darts_tfm_arguments() to get recommendations.  
    # min_max_scale==True will scale the covariates 0.0 to +1.0

    # IMPORTANT:  It is not strictly necessary to slice covariates before using them with .fit() and .predict() because Darts will
    #             handle the slicing.  However, if the past/future covariates you pass to .fit() and/or .predict() don't have the
    #             correct time span / length, then errors will be raised.  This function makes sure the covariates have the required
    #             time spans.  

    if verbose: print("\ndarts_get_tfm_covariates()..")

    from darts import TimeSeries

    from darts.dataprocessing.transformers.scaler import Scaler
    scaler = Scaler()  # default uses sklearn's MinMaxScaler

    from datetime import datetime, timedelta

    if not isinstance(min_max_scale, bool): raise Exception("Value Error: min_max_scale must be boolean type with value of True or False")
    if not isinstance(verbose, bool): raise Exception("Value Error: verbose must be boolean type with value of True or False")

    # Validate any series passed
    if train is None or not isinstance(train, TimeSeries): raise Exception("Value Error: train is not a Darts TimeSeries")
    if not val_series is None and not isinstance(val_series, TimeSeries): raise Exception("Value Error: val_series is not a Darts TimeSeries")
    if test is None or not isinstance(test, TimeSeries): raise Exception("Value Error: test is not a Darts TimeSeries")
    if not pred_input is None and not isinstance(pred_input, TimeSeries): raise Exception("Value Error: pred_input is not a Darts TimeSeries")
    if pred_steps is None or not isinstance(pred_steps, int): raise Exception("Value Error: pred_steps is not None or an integer")

    if val_series is None and not val_past_covariates is None: raise Exception("Value Error: val_past_series must be None if val_series is None")
    if val_series is None and not val_future_covariates is None: raise Exception("Value Error: val_future_covariates must be None if val_series is None")
    if not val_series is None and not past_covariates is None and val_past_covariates is None: raise Exception("Value Error: val_past_series must NOT be None if val_series is not None and past_covariates is not None")
    if not val_series is None and not future_covariates is None and val_future_covariates is None: raise Exception("Value Error: val_future_covariates must NOT be None if val_series is not None and future_covariates is not None")

    # Convert any arguments with value None to list so len() doesn't throw error
    val_series = [] if not val_series else val_series
    pred_input = [] if not pred_input else pred_input

    # past_covariates, val_past_covariates, future_covariates, val_future_covariates, static_covariates
    if not past_covariates is None and not isinstance(past_covariates, TimeSeries): raise Exception("Value Error: past_covariates is not a Darts TimeSeries")
    if not val_past_covariates is None and not isinstance(val_past_covariates, TimeSeries): raise Exception("Value Error: val_past_covariates is not a Darts TimeSeries")
    if not future_covariates is None and not isinstance(future_covariates, TimeSeries): raise Exception("Value Error: future_covariates is not a Darts TimeSeries")
    if not val_future_covariates is None and not isinstance(val_future_covariates, TimeSeries): raise Exception("Value Error: val_future_covariates is not a Darts TimeSeries")

    if not isinstance(output_chunk_length, int): raise Exception("Value Error: output_chunk_length_max must be int type >= 1")
    if output_chunk_length < 1: raise Exception("Value Error: output_chunk_length must be >= 1 ", output_chunk_length)

    if verbose: print("output_chunk_length:", output_chunk_length)
    if pred_steps <= output_chunk_length:
        # n <= output_chunk_length
        if verbose: print("n <= output_chunk_length \tWARNING: Setting n <= output_chunk_length prevents auto-regression for Torch Forecasting Models\t", pred_steps, "<=", output_chunk_length,"\n")
    else:
        # n > output_chunk_length
        if verbose: print("n > output_chunk_length (permits auto-regression)")

    if past_covariates is None:
        val_past_covariates = None
    else:
        # Create a Darts covariate past timeseries from ts.
        # The exact length requirements vary by if n <= output_chunk_length, or if n > output_chunk_length.
        # Slice past_covariates to the minimum length required for .predict()
        if pred_steps <= output_chunk_length:
            # n <= output_chunk_length
            past_covariates = past_covariates.slice_n_points_after(ts.time_index.min(), len(train)+len(val_series)+len(pred_input))
            if verbose: print("The past covariates must end no earlier than time step ", past_covariates.time_index.max(), "\tlength: ", len(past_covariates.time_index))
        else:
            # n > output_chunk_length
            past_covariates = past_covariates.slice_n_points_after(ts.time_index.min(), len(train)+len(val_series)+len(pred_input)+(pred_steps-output_chunk_length))
            if verbose: print("The past covariates must end no earlier than time step ", past_covariates.time_index.max(), "\tlength: ", len(past_covariates.time_index))        
        
        # Create val_past_covariates from past_covariates before fitting past_covariates with the scaler.
        if not len(val_series)==0:  val_past_covariates = past_covariates.copy()
        # Fit the scaler to the covariate associated with the series 'train'.
        if min_max_scale == True: 
            past_covariates = scaler.fit_transform(past_covariates)
            # Another approach:
            #scaler.fit(past_covariates)
            #past_covariates = scaler.transform(past_covariates)
        if verbose: print("past_covariates: ", past_covariates.time_index.min(), " ..", past_covariates.time_index.max(), "\tlength: ", len(past_covariates.time_index), "") 
        #Scale val_past_covariates if appropriate
        if not len(val_series)==0 and min_max_scale == True:  val_past_covariates = scaler.transform(val_past_covariates)

    # # Create a Darts covariate future timeseries.
    if future_covariates is None:
        val_future_covariates = None
    else:
        if pred_steps <= output_chunk_length:
            # n <= output_chunk_length
            future_covariates = future_covariates.slice_n_points_after(train.time_index.min(), len(train)+len(val_series)+len(pred_input)+output_chunk_length) 
            if len(train)+len(val_series)+len(pred_input)+output_chunk_length >= len(ts):
                print("\nWARNING: the spare time span after the forecast period n (pred_steps) is insufficient!  Reduce the size of pred_input, n (pred_steps), or output_chunk_length by at least " + str((len(train)+len(val_series)+len(pred_input)+output_chunk_length+1)-len(ts)) + " rows")
        else:
            # n > output_chunk_length
            future_covariates = future_covariates.slice_n_points_after(train.time_index.min(), len(train)+len(val_series)+len(pred_input)+pred_steps)
        
        # Create val_future_covariates from past_covariates before fitting future_covariates with the scaler.
        if not len(val_series)==0:  val_future_covariates = future_covariates.copy()
        # Fit the scaler to the covariate associated with the series 'train'.
        if min_max_scale == True: 
            future_covariates = scaler.fit_transform(future_covariates)
            # Another approach:
            #scaler.fit(future_covariates)
            #future_covariates = scaler.transform(future_covariates)
        if verbose: print("future_covariates: ", future_covariates.time_index.min(), " ..", future_covariates.time_index.max(), "\tlength: ", len(future_covariates.time_index), "") 
        #Scale val_future_covariates if appropriate
        if not len(val_series)==0 and min_max_scale == True:  val_future_covariates = scaler.transform(val_future_covariates)

    return past_covariates, val_past_covariates, future_covariates, val_future_covariates


def slice_tfm_covariates_for_trained_model(model=None, ts=None, pred_input=None, pred_steps=None, past_covariates=None, future_covariates=None, min_max_scale=True, verbose=False):
    # Returns sliced past/future covariates from the series 'ts' for the trained model 'model'
    # to be used for a new prediction specified by pred_input and pred_steps. 
    # Returns static_covariates from the trained model if previously used for fitting the model. 
    # NOTE: Do not use this function for model training, instead use: slice_tfm_covariates_for_model_training().
    # Validates the length of pred_input and pred_steps for use with the passed trained model. 
    # The trained model attributes 'uses_past_covariates', 'uses_future_covariates', and
    # 'uses_static_covariates are used to determine what covariates are returned. When not 
    # applicable, then None is returned.
    # min_max_scale==True will scale the covariates 0.0 to +1.0

    # IMPORTANT:  It is not strictly necessary to slice covariates before using them with .fit() and .predict() because Darts will
    #             handle the slicing.  However, if the past/future covariates you pass to .fit() and/or .predict() don't have the
    #             correct time span / length, then errors will be raised.  This function makes sure the covariates have the required
    #             time spans.  


    from darts import TimeSeries
    from darts.dataprocessing.transformers.scaler import Scaler
    scaler = Scaler()  # default uses sklearn's MinMaxScaler

    import pandas as pd
    from datetime import datetime, timedelta

    if model is None: raise Exception("You must pass a Darts model object")
    if not isinstance(min_max_scale, bool): raise Exception("Value Error: min_max_scale must be boolean type with value of True or False")
    if not isinstance(verbose, bool): raise Exception("Value Error: verbose must be boolean type with value of True or False")

    # Validate any series passed
    if ts is None or not isinstance(ts, TimeSeries): raise Exception("Value Error: train is not a Darts TimeSeries")
    if pred_input is None or not isinstance(ts, TimeSeries): raise Exception("Value Error: pred_input is not a Darts TimeSeries")
    if pred_steps is None or not isinstance(pred_steps, int): raise Exception("Value Error: pred_steps must be an integer for the number of series rows in the future for the prediction.")
    if not pred_steps is None and pred_steps < 1: raise Exception("Value Error: pred_steps must be >= 1 or None")
    if not past_covariates is None and not isinstance(past_covariates, TimeSeries): raise Exception("Value Error: past_covariates is not a Darts TimeSeries")
    if not future_covariates is None and not isinstance(future_covariates, TimeSeries): raise Exception("Value Error: future_covariates is not a Darts TimeSeries")
    if model.training_series is None: raise Exception("The model has no training series associated with it.  Train the model first.")

    if verbose: print("slice_tfm_covariates_for_trained_model()..")

    if verbose: print("\tmodel_name:", model.model_name)
    #if verbose: print("\tmodel_created:", model.model_created)                                # True
    if verbose: print("\tinput_chunk_length: ", model.input_chunk_length)               
    if verbose: print("\toutput_chunk_length:", model.output_chunk_length)              
    if 'kernel_size' in model.model_params: print("\tkernel_size:", model.model_params['kernel_size'])                    # 25
    #if verbose: print("\tfirst_prediction_index:", model.first_prediction_index)              # 0
    #if verbose: print("\textreme_lags: ", model.extreme_lags)                                 # 
    # (min target lag, max target lag, min past covariate lag, max past covariate lag, min future covariate lag, max future covariate lag). 
    # If 0 is the index of the first prediction, then all lags are relative to this index.        
    #if verbose: print("\tmodel_params:", model.model_params)                                  # OrderedDict([('shared_weights', False), ('kernel_size', 25), ('const_init', True), ('use_static_covariates', True), ('input_chunk_length', 24), ('output_chunk_length', 14)])
    #if verbose: print("\ttrainer_params:", model.trainer_params)                              # {'logger': False, 'max_epochs': 100, 'check_val_every_n_epoch': 1, 'enable_checkpointing': False, 'callbacks': [], 'precision': '32-true'}
    #if verbose: print("\tmin_train_samples:", model.min_train_samples)                        # 1
    #if verbose: print("\tmin_train_series_length:", model.min_train_series_length)            # 35 
    #if verbose: print("\toutput_dim:", model.output_dim)                                      # 1         1=univariate, >1=multivariate ??
    if verbose: print("\tmodel.training_series: ", model.training_series.time_index.min(), " ..", model.training_series.time_index.max(), "\tlength: ", len(model.training_series.time_index))  
    #if not model.trainer is None: print("\tmodel.trainer: ", model.trainer.time_index.min(), " ..", model.trainer.time_index.max(), "\tlength: ", len(model.trainer.time_index))  
    if not model.past_covariate_series is None: print("\tmodel.past_covariate_series: ", model.past_covariate_series.time_index.min(), " ..", model.past_covariate_series.time_index.max(), "\tlength: ", len(model.past_covariate_series.time_index))      
    if not model.future_covariate_series is None: print("\tmodel.future_covariate_series: ", model.future_covariate_series.time_index.min(), " ..", model.future_covariate_series.time_index.max(), "\tlength: ", len(model.future_covariate_series.time_index))  
    if model.uses_past_covariates == True: print("\tThis model uses past covariates and therfore they must also be included with any predictions.")
    if model.uses_future_covariates == True: print("\tThis model uses future covariates and therfore they must also be included with any predictions.")
    if model.uses_static_covariates == True: print("\tThis model uses static covariates and therfore they must also be included with any predictions.")
    if verbose: print("\tNOTE: The minimum length of any target series passed to .predict() using this trained model must have a length >= input_chunk_length: ", model.input_chunk_length)               
    if verbose: print("\n")

    if verbose: print("ts: ", ts.time_index.min(), " ..", ts.time_index.max(), "\tlength: ", len(ts.time_index))  

    if verbose: print("\npred_input: ", pred_input.time_index.min(), " ..", pred_input.time_index.max(), "\tlength: ", len(pred_input.time_index))  
    # Validate the length of pred_input/test against input_chunk_length
    if len(pred_input) < model.input_chunk_length: raise Exception("Value Error: The length of pred_input (" + str(len(pred_input)) + ") passed to .predict() must be >= to input_chunk_length of " + str(model.input_chunk_length))
    if verbose: print("NOTE: The length of pred_input (" + str(len(pred_input)) + ") passed to .predict() must be >= to input_chunk_length of " + str(model.input_chunk_length) + "\n")

    if isinstance(ts.time_index, pd.DatetimeIndex):
        t_pred_steps = pred_input.time_index.max() + timedelta(days=pred_steps)
    else:
        t_pred_steps = pred_input.time_index.max() + pred_steps
    if t_pred_steps > ts.time_index.max(): raise Exception("Value Error: pred_steps is too large (or pred_input starts too late) and establishes a pred_steps end date beyond the length of the source series ts.  Max pred_steps is " + str(ts.time_index.max()-pred_input.time_index.max()))
    if isinstance(ts.time_index, pd.DatetimeIndex):
        pred_steps_ts = ts.slice(pred_input.time_index.max() + timedelta(days=1), t_pred_steps)
    else:
        pred_steps_ts = ts.slice(pred_input.time_index.max() + 1, t_pred_steps)
    if verbose: print("pred_steps", str(pred_steps_ts.time_index.min()), str(pred_steps_ts.time_index.max()), "\tlength:", len(pred_steps_ts))
    if pred_steps > model.output_chunk_length:
        if verbose: print("n > output_chunk_length")
    else:
        if verbose: print("n <= output_chunk_length")

    if model.uses_past_covariates == False:
        past_covariates = None
    elif past_covariates is None:
        raise Exception("The model uses past covariates, but a series to be sliced for past covariates was not provided.")
    else:
        # model.uses_past_covariates == True
        # Create a covariant series for input to .predict()
        # The past covariant series (past_covariates) index and length requirements are completely independent of the 
        # trained model in terms of index time span (date values).
        # The only requirement for past_covariates is it's length:
        #   if n (pred_steps) > model.output_chunk_length, then the length must be >= len(pred_input)+(pred_steps-model.output_chunk_length)
        #   if n (pred_steps) <= model.output_chunk_length, then the length must be >= len(pred_input)
        # min/max scale future_covariates using the scaler fit on series 'train'
        if min_max_scale == True: past_covariates = scaler.fit_transform(past_covariates)
        if pred_steps > model.output_chunk_length:
            # n > output_chunk_length
            past_covariates = past_covariates.slice_n_points_after(pred_input.time_index.min(), len(pred_input)+(pred_steps-model.output_chunk_length))
            if len(past_covariates) < len(pred_input)+(pred_steps-model.output_chunk_length): raise Exception("ERROR: the length of past_covariates of " + str(len(past_covariates)) + " < " + str(len(pred_input)+(pred_steps-model.output_chunk_length)) + " len(pred_input)+(pred_steps-model.output_chunk_length).  Make sure the source of past_covariates has sufficient length.")
        else:
            # n <= output_chunk_length
            past_covariates = past_covariates.slice_n_points_after(pred_input.time_index.min(), len(pred_input)) 
            if len(past_covariates) < len(pred_input): raise Exception("ERROR: the length of past_covariates " + str(len(past_covariates)) + " must be >= len(pred_input) " + str(len(pred_input)))
        if verbose: print("\npast_covariates: ", past_covariates.time_index.min(), " ..", past_covariates.time_index.max(), "\tlength: ", len(past_covariates.time_index))  

    if model.uses_future_covariates == False:
        future_covariates = None
    elif future_covariates is None:
        raise Exception("The model uses future covariates, but a series to be sliced for future covariates was not provided.")
    else:
        # model.uses_future_covariates == True
        # Create the future covariate for input to .predict()
        # Note that the index of future_covariates must start before 
        # pred_input and end after pred_input as coded below.
        # min/max scale future_covariates using the scaler fit on series 'train'
        if min_max_scale == True: future_covariates = scaler.fit_transform(future_covariates)
        if pred_steps > model.output_chunk_length:
            # n > model.output_chunk_length
            # future_covariates index must start (model.output_chunk_length - 1) steps before the end of the series pred_input.
            # NOTE:  Below calculation is according to Darts documentation, BUT it doesn't work.  Use model.input_chunk_length in place of model.output_chunk_length
            #t_start = pred_input.time_index.max() + timedelta(days=model.output_chunk_length*(-1)-1)
            if isinstance(ts.time_index, pd.DatetimeIndex):
                t_start = pred_input.time_index.max() + timedelta(days=model.input_chunk_length*(-1)+1)
                t_end = pred_input.time_index.max() + timedelta(days=pred_steps)        # NOTE: Darts documentation says n-model.output_chunk_length but that doesn't work!
            else:
                t_start = pred_input.time_index.max() + (model.input_chunk_length*(-1)+1)
                t_end = pred_input.time_index.max() + pred_steps +1       # NOTE: Darts documentation says n-model.output_chunk_length but that doesn't work!
            #print("Note: future_covariates must start no later than:", t_start, "(model.output_chunk_length-1 days) and continue at least pred_steps=" + str(pred_steps) + " steps after the end of pred_input to ", t_end)    
            future_covariates = future_covariates.slice(t_start, t_end)      
        else:
            # n <= model.output_chunk_length
            # future_covariates index must start (model.output_chunk_length + 1) steps before the end of the series pred_input.
            # NOTE:  Below calculation is according to Darts documentation, BUT it doesn't work.  Use model.input_chunk_length in place of model.output_chunk_length
            #start_t = pred_input.time_index.max()+timedelta(days=model.output_chunk_length*(-1)-1)
            if isinstance(ts.time_index, pd.DatetimeIndex):
                start_t = pred_input.time_index.max() + timedelta(days=model.input_chunk_length*(-1)+1)
            else:
                start_t = pred_input.time_index.max() + (model.input_chunk_length*(-1)+1)
            #print("Note: future_covariates must start no later than:", start_t, "(model.input_chunk_length+1 days) and have a length of at least len(pred_input)+model.output_chunk_length")    
            future_covariates = future_covariates.slice_n_points_after(start_t, len(pred_input)+model.output_chunk_length)      
        if verbose: print("\nfuture_covariates: ", future_covariates.time_index.min(), " ..", future_covariates.time_index.max(), "\tlength: ", len(future_covariates.time_index))  

    static_covariates = None
    if model.uses_static_covariates == True:
        static_covariates = model.static_covariates

    return past_covariates, future_covariates, static_covariates


# ---------------------------------------------------------------------------
# Darts Local Forecasting Models (LFM)



if __name__ == '__main__':
    pass


    # ---------------------------------------------------------------------------
    # Demonstrate using sine_gaussian_noise_covariate() to create univariate, 
    # multivariate, and covariate signals with no trend.
    # Demo plotting options with plt_darts_ts_stacked(). 

    """
    from darts import TimeSeries
    import matplotlib.pyplot as plt

    # Get a data set with 3x signals (multivariate) & a covariate (sine wave).
    df = sine_gaussian_noise_covariate(add_trend=False, include_covariate=True, multivariate=True)
    #print(df)
    #component        ts1       ts2       ts3  covariate
    #Date
    #2000-01-01  0.357390  1.500000 -0.972272   0.500000
    #2000-01-02 -0.021171  1.528882 -1.099552   0.562667
    #2000-01-03  0.786181  1.822958 -0.481920   0.624345
    #...              ...       ...       ...        ...
    #2001-02-03  0.099003  0.641921 -1.090199   0.437333

    # Get three Darts univariate TimeSeries from the first 3 columns in df & built a multivariate TimeSeries
    ts = TimeSeries.from_series(pd_series=df.iloc[:,0]) 
    ts2 = TimeSeries.from_series(pd_series=df.iloc[:,1]) 
    ts3 = TimeSeries.from_series(pd_series=df.iloc[:,2]) 
    ts_mv = ts.stack(ts2)
    ts_mv = ts_mv.stack(ts3)
    del ts2, ts3
    # ts is a univariate series from the first column in df
    # ts_mv is a multivariate TimeSeries with 3x components (columns) from the first 3 columns in df

    # Get a Darts univariate covariate timeseries from the 4nd column in df,
    covariate = TimeSeries.from_series(pd_series=df.iloc[:,3]) 

    # Plot ts, covariate, ts_mv using plt_darts_ts_stacked()

    plt_darts_ts_stacked(ts, title="ts univariate")

    plt_darts_ts_stacked(ts_mv, title="ts_mv multivariate")

    plt_darts_ts_stacked(ts=[ts, covariate], labels=['ts', 'covariate'], title="List of 2x Univariates")

    plt_darts_ts_stacked(ts=[ts_mv,covariate], labels=['ts_mv','covariate'], title="List of 1x Multivariate + 1x Univariate")

    plt_darts_ts_stacked(ts=[covariate,ts_mv], labels=['covariate','ts_mv'], title="List of  1x Univariate + 1x Multivariate")

    plt_darts_ts_stacked(ts=[ts_mv,ts, covariate], labels=['ts_mv','ts','covariate'], title="List of 1x Multivariate + 1x Univariate + 1x Univariate")
    """

    # ---------------------------------------------------------------------------
    # get_darts_model_splits()


    """
    from darts import TimeSeries
    #from medium_darts import sine_gaussian_noise_covariate, get_darts_model_splits

    df = sine_gaussian_noise_covariate(add_trend=False, include_covariate=True)
    df.reset_index(inplace=True, drop=True)     # Convert index from datetime to integer

    # Get a Darts univariate timeseries from the first column in df,
    ts = TimeSeries.from_series(pd_series=df.iloc[:,0]) 

    # Split the TimeSeries data into train, val, test, pred_input, pred_steps
    train, val, test, pred_input, pred_steps = get_darts_model_splits(ts=ts, train=0.7, val_series=0.1, test=0.2, pred_input=0.3, pred_steps=0.2, min_max_scale=True, plot=False)

    #              series|              index start|                index end|         length|          width
    #                  ts|      2000-01-01 00:00:00|      2001-02-03 00:00:00|            400|              1
    #               train|      2000-01-01 00:00:00|      2000-10-06 00:00:00|            280|              1
    #          val_series|      2000-10-07 00:00:00|      2000-11-15 00:00:00|             40|              1
    #                test|      2000-11-16 00:00:00|      2001-02-03 00:00:00|             80|              1
    #          pred_input|      2000-11-16 00:00:00|      2000-12-09 00:00:00|             24|              1
    #          pred_steps|      2000-12-10 00:00:00|      2000-12-25 00:00:00|             16|              1
    #   future covariates|      2000-12-26 00:00:00|      2001-02-03 00:00:00|             40|              1
    
    """


    # ---------------------------------------------------------------------------
    # Optimize input_chunk_length & output_chunk_length for all Torch Forecasting 
    # Models that support past, future, & static covariates and multivariate series
    # against the sine_gaussian_noise_covariate() dataset using the custom functions 
    # below, save the trained models, and then plot the results for the best model.
    #   get_darts_model_splits()
    #   get_darts_tfm_arguments()
    #   get_darts_tfm_arguments_optimized()
    #   slice_tfm_covariates_for_model_training()
    #   plt_model_training()


    def train_tfm_that_support_past_future_static_covariates():

        from darts import TimeSeries
        from darts.metrics import rmse
        from darts.dataprocessing.transformers.scaler import Scaler
        import torch

        from pathlib import Path
        from operator import itemgetter
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np

        import warnings
        warnings.filterwarnings("ignore")
        import logging
        logging.disable(logging.CRITICAL)
        from time_series_data_sets import sine_gaussian_noise_covariate

        # for reproducibility
        # https://vandurajan91.medium.com/random-seeds-and-reproducible-results-in-pytorch-211620301eba
        # https://pytorch.org/docs/stable/notes/randomness.html
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        import random
        np.random.seed(1)

        # Revise the path below for your system.  Below saves to folder 'data' in the parent of the current working folder.
        # Later in the script, the model filename will be defined using model_save_path with .joinpath() just prior to saving the trained model. 
        model_save_path = Path(Path.cwd().parent).joinpath('data/')
        if not model_save_path.exists(): raise Exception("Folder specified by 'path_file' doesn't exist: "+ str(model_save_path))

        scaler = Scaler()  # default uses sklearn's MinMaxScaler.  

        # Get a data set with 3x signals & a covariate (sine wave).
        df = sine_gaussian_noise_covariate(add_trend=False, include_covariate=True, multivariate=True)

        # Get three Darts univariate TimeSeries from the first 3 columns in df & built a multivariate TimeSeries
        ts1 = TimeSeries.from_series(pd_series=df.iloc[:,0]) 
        ts2 = TimeSeries.from_series(pd_series=df.iloc[:,1]) 
        ts3 = TimeSeries.from_series(pd_series=df.iloc[:,2]) 
        ts = ts1.stack(ts2)
        ts = ts.stack(ts3)
        del ts1, ts2, ts3
        # ts is now a multivariate TimeSeries with 3x components (columns)

        # Set min_max_scale = True to min/max scale the data to range of 0.0 to +1.0
        min_max_scale = True

        # Define static covariates
        # Note: Add any static covariates to series 'ts' BEFORE splitting 'ts' into train, val, test, pred_input..
        #static_covariates = None
        static_covariates = pd.DataFrame(data={"cont": [0, 2, 1], "cat": ["a", "c", "b"]})
        if not static_covariates is None:
            # Add the static covariates to the TimeSeries
            ts = ts.with_static_covariates(static_covariates)
            print(ts.static_covariates)
            
        # Split ts into train, val_series, test, pred_input, and get the pred_steps.  Scale all data to 0.0 to +1.0
        train, val, test, pred_input, pred_steps = get_darts_model_splits(ts=ts, train=0.6, val_series=0.1, test=0.3, pred_input=0.45, pred_steps=0.4, min_max_scale=min_max_scale, plot=False)

        # Define any past/future covariates
        past_covariates = None
        future_covariates = None
        val_past_covariates = None
        val_future_covariates = None
        # Create the raw full length series for the covariates from the 4th column of df
        past_covariates = TimeSeries.from_series(pd_series=df.iloc[:,3]) 
        future_covariates = TimeSeries.from_series(pd_series=df.iloc[:,3]) 
        # Create a static covariate
        if not val is None:
            if not past_covariates is None: val_past_covariates = past_covariates.copy()
            if not future_covariates is None: val_future_covariates = future_covariates.copy()

        # Define the models to be processed
        from darts.models import (TFTModel,DLinearModel,NLinearModel,TiDEModel)
        # Try each of the Torch Forecasting Models that support past, future, & static covariates and multivariates.
        # Consult the the following table to see what models to choose:  https://github.com/unit8co/darts?tab=readme-ov-file#forecasting-models
        models = {
            'TFTModel': TFTModel,      
            'DLinearModel': DLinearModel, 
            'NLinearModel': NLinearModel, 
            'TiDEModel': TiDEModel,
        }

        results = []
        for m in models:
            print("\n===========================================================================")
            print("Model: ", m)

            # Get input_chunk_length/output_chunk_length recommendations
            input_chunk_length, output_chunk_length, input_chunk_length_max, output_chunk_length_max, sum_chunk_length_limit, model_specific = get_darts_tfm_arguments(m, train, val, test, pred_input, pred_steps, verbose=True)
            
            # Optimize input_chunk_length and output_chunk_length
            input_chunk_length, output_chunk_length, metric_result, t_elapsed = get_darts_tfm_arguments_optimized(m, models[m], 
                                                                                                                    input_chunk_length_max, output_chunk_length_max, sum_chunk_length_limit,
                                                                                                                    ts=ts, train=train, test=test, val_series=val, pred_input=pred_input, pred_steps=pred_steps, 
                                                                                                                    past_covariates=past_covariates, val_past_covariates=val_past_covariates, future_covariates=future_covariates, val_future_covariates=val_future_covariates,
                                                                                                                    metric=rmse, min_max_scale=min_max_scale,
                                                                                                                    coarse_only=False, verbose=True)
            
            print("Optimization of input_chunk_length/output_chunk_length " + str(input_chunk_length) + "/" + str(output_chunk_length) + " took", t_elapsed, "sec\n")

            # TRAIN, FIT, PREDICT model with optimized input_chunk_length, output_chunk_length
            print("Train, fit, predict model " + m + " with optimized input_chunk_length: ", input_chunk_length, "\toutput_chunk_length: ", output_chunk_length)
            
            # Slice the covariates to the required size for training..
            cv_past, cv_val_past, cv_future, cv_val_future = slice_tfm_covariates_for_model_training(ts, train, val, test, pred_input, pred_steps, min_max_scale=min_max_scale, output_chunk_length=output_chunk_length, past_covariates=past_covariates, val_past_covariates=val_past_covariates, future_covariates=future_covariates, val_future_covariates=val_future_covariates, verbose=True)

            model = models[m](input_chunk_length=input_chunk_length, output_chunk_length=output_chunk_length)     
            
            model.fit(series=train, val_series=val, val_past_covariates=cv_val_past, val_future_covariates=cv_val_future, past_covariates=cv_past, future_covariates=cv_future)

            pred = model.predict(n=pred_steps, series=pred_input, past_covariates=cv_past, future_covariates=cv_future)  # defaults to series=train when pred_input==None
            print(m, "pred: ", pred.time_index.min(), " ..", pred.time_index.max(), "\tlength: ", len(pred.time_index))  
            # Create an original / full scale version of the prediction
            train = scaler.fit_transform(model.training_series)
            pred_fs = scaler.inverse_transform(pred)        

            # Save the model that you just trained.
            # https://unit8co.github.io/darts/userguide/forecasting_overview.html#saving-and-loading-models
            # Note that the .ckpt extension is omitted purposely.
            # Darts model.save() method doesn't accept a pathlib object, only a string for the path/filename.
            path_file = model_save_path.joinpath(m + '_pc_fc_sc.pt')       # '_pc' = past covariate, _fc future covariate, _sc static covariate
            print("Saving model " + m + " to: " + str(path_file))
            model.save(str(path_file))

            # Calculate the root mean square error (RMSE)
            ground_truth = ts.slice_intersect(pred)
            metric_rmse = rmse(ground_truth, pred)
            print(m + " RMSE: ", round(metric_rmse,5))   
            
            # Optionally plot the input TimeSeries and the results for each model
            title = m + "  RMSE: " + str(round(metric_rmse,5))
            #plt_model_training(train, val, test, pred_input, pred_steps, pred, title)

            # Optionally plot in original / full scale 
            #plt_darts_ts_stacked(ts=[ground_truth, pred_fs], labels=['ground truth','prediction'], title=title)

            # Store the model performance & optimized configuration results
            results.append([metric_rmse, m, title, input_chunk_length, output_chunk_length, model, path_file])

        # Show the results sorted by RMSE
        results = sorted(results, key=itemgetter(0))
        template = "{0:>20}|{1:>20}|{2:>20}|{3:>20}|" # column widths: 20, 20, 20    '>' right justify
        print(template.format("RMSE", "model","input_chunk_length","output_chunk_length")) # header
        for result in results:
            print(template.format(result[0], result[1], result[3], result[4]))
        #                RMSE|               model|  input_chunk_length| output_chunk_length|
        #  0.8240864568989227|        NLinearModel|                  18|                  14|
        #  0.8308660678674741|           TiDEModel|                  12|                   1|
        #  0.8385293004862168|        DLinearModel|                  20|                   4|
        #  0.8779658714240849|            TFTModel|                  20|                   6|
            
        # Read the saved model for the best result (lowest RMSE) and plot the result.
        m = results[0][1]
        model_class = results[0][5]
        metric_rmse = results[0][0]
        input_chunk_length = results[0][3]
        output_chunk_length = results[0][4]
        cv_past, cv_val_past, cv_future, cv_val_future = slice_tfm_covariates_for_model_training(ts, train, val, test, pred_input, pred_steps, min_max_scale=min_max_scale, output_chunk_length=output_chunk_length, past_covariates=cv_past, val_past_covariates=cv_val_past, future_covariates=cv_future, val_future_covariates=cv_val_future, verbose=False)
        path_file = model_save_path.joinpath(m + '_pc_fc_sc.pt')       
        model = model_class.load(str(path_file))
        pred = model.predict(n=pred_steps, series=pred_input)  # defaults to series=train when pred_input==None        
        print("pred: ", pred.time_index.min(), " ..", pred.time_index.max(), "\tlength: ", len(pred.time_index))  
        title = "BEST MODEL: " + m + "  RMSE: " + str(round(metric_rmse,5))
        plt_model_training(train, val, test, pred_input, pred_steps, pred, title, cv_past, cv_future)

        # Show all of the saved trained models
        print("\nSummary of all saved models:")
        for m in models:
            path_file = model_save_path.joinpath(m + '_pc_fc_sc')
            if path_file.exists(): print(m + "\t\t" + str(path_file))
        # To read a trained model later:
        #from pathlib import Path
        #from darts.models import (DLinearModel)
        #path_file = Path(Path.cwd().parent).joinpath('data/DLinearModel_pc_fc_sc.pt')
        #model = DLinearModel.load(str(path_file))


    # WARNING:  Validate the model save path before running this script.
    #           Optimization can take several hours.
    
    #train_tfm_that_support_past_future_static_covariates()


    # ---------------------------------------------------------------------------
    # Run all past, future, & static covariate trained Torch Forecasting Models
    # generated from the prior step (train_tfm_that_support_past_future_static_covariates) 
    # against a new series.

    def test_all_past_future_static_covariate_trained_tfm_against_new_series():

        from darts import TimeSeries
        from darts.metrics import rmse, rmsle
        from darts.dataprocessing.transformers.scaler import Scaler
        from darts.dataprocessing.transformers import StaticCovariatesTransformer
        import torch

        from pathlib import Path
        from operator import itemgetter
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        import warnings
        warnings.filterwarnings("ignore")
        import logging
        logging.disable(logging.CRITICAL)

        # for reproducibility
        # https://vandurajan91.medium.com/random-seeds-and-reproducible-results-in-pytorch-211620301eba
        # https://pytorch.org/docs/stable/notes/randomness.html
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        import random
        np.random.seed(1)

        # Revise the path below for your system.  Below saves to folder 'data' in the parent of the current working folder.
        # Later in the script, the model filename will be defined using model_save_path with .joinpath() just prior to saving the trained model. 
        model_save_path = Path(Path.cwd().parent).joinpath('data/')
        if not model_save_path.exists(): raise Exception("Folder specified by 'path_file' doesn't exist: "+ str(model_save_path))

        scaler = Scaler()  # default uses sklearn's MinMaxScaler.  Does NOT support multivariate series
        transformer = StaticCovariatesTransformer()

        # Get a data set with 3x signals & a covariate (sine wave) with a starting data different than when the model was trained.
        df = sine_gaussian_noise_covariate(add_trend=False, include_covariate=True, multivariate=True, start=pd.Timestamp(year=2005, month=12, day=27))

        # Get three Darts univariate TimeSeries from the first 3 columns in df & built a multivariate TimeSeries.
        # Scale each to range of 0.0 to +1.0
        ts1 = TimeSeries.from_series(pd_series=df.iloc[:,0]) 
        ts1 = scaler.fit_transform(ts1)
        ts2 = TimeSeries.from_series(pd_series=df.iloc[:,1]) 
        ts2 = scaler.fit_transform(ts2)
        ts3 = TimeSeries.from_series(pd_series=df.iloc[:,2]) 
        ts3 = scaler.fit_transform(ts3)
        ts = ts1.stack(ts2)
        ts = ts.stack(ts3)
        del ts1, ts2, ts3
        # ts is now a multivariate TimeSeries with 3x components (columns)

        # Add the static covariates to ts
        # Note: this must be done before series pred_input is sliced from ts.
        static_covariates = pd.DataFrame(data={"cont": [0, 2, 1], "cat": ["a", "c", "b"]})
        # Add the static covariates to series 'ts'
        ts = ts.with_static_covariates(static_covariates)
        # Note: TiDEModel can only interpret numeric static covariate data, so use .fit_transform()
        # to match what was done by get_darts_model_splits() during model training.
        # [[0.0 'a'],[2.0 'c'],[1.0 'b']]   =>  [[0.  0. ],[1.  2. ],[0.5 1. ]]
        ts = transformer.fit_transform(ts)

        # Define the models to be processed
        from darts.models import (TFTModel,DLinearModel,NLinearModel,TiDEModel)
        # Try each of the Torch Forecasting Models that support past, future, & static covariates and multivariates.
        # Consult the the following table to see what models to choose:  https://github.com/unit8co/darts?tab=readme-ov-file#forecasting-models
        models = {
            'TiDEModel': TiDEModel,
            'DLinearModel': DLinearModel, 
            'NLinearModel': NLinearModel, 
            'TFTModel': TFTModel,      
        }

        results = []
        for m in models:
            print("\n===========================================================================")
            print("Trained model: ", m)

            # Read the previously trained and saved model.
            path_file = model_save_path.joinpath(m + '_pc_fc_sc.pt')
            if not path_file.exists(): raise Exception("Saved model .pt was not found: " + str(path_file))
            model = models[m].load(str(path_file))

            # Create a series pred_input from ts for input to .predict(series=pred_input) with a length > input_chunk_length
            # start at row 50 and end with a length of 100 rows
            pred_input = ts[50:50+100]            
            # Slice by specific date values                            
            #pred_input = ts.slice(pd.Timestamp(year=2006, month=6, day=1), pd.Timestamp(year=2006, month=9, day=8))

            # Define the the number of steps (series rows) or n beyond the end of pred_input for the prediction.
            pred_steps = 150

            # Define the raw past/future covariates
            past_covariates = TimeSeries.from_series(pd_series=df.iloc[:,1]) 
            past_covariates = TimeSeries.from_series(pd_series=df.iloc[:,3]) 
            future_covariates = TimeSeries.from_series(pd_series=df.iloc[:,3]) 

            # Slice the covariates to the minimim start/end required for the model .predict() method.  
            cv_past, cv_future, cv_static = slice_tfm_covariates_for_trained_model(model, ts, pred_input, pred_steps, past_covariates, future_covariates, min_max_scale=True, verbose=True)

            pred = model.predict(n=pred_steps, series=pred_input, past_covariates=cv_past, future_covariates=cv_future)  # defaults to series=train when pred_input==None
            print("\npred: ", pred.time_index.min(), " ..", pred.time_index.max(), "\tlength: ", len(pred.time_index))  

            # Calculate the root mean square error (RMSE)
            metric_rmse = rmse(ts.slice_intersect(pred), pred)
            print(m + " RMSE: ", round(metric_rmse,5))   
            results.append([metric_rmse, m, model.input_chunk_length, model.output_chunk_length])
            
            # Plot
            title = m + " RMSE: " + str(round(metric_rmse,5))
            plt_model_trained(ts, cv_past, cv_future, pred_input, pred, title)

        # Summarize the results by best RMSE first
        print("\n")
        results = sorted(results, key=itemgetter(0))
        template = "{0:>20}|{1:>20}|{2:>20}|{3:>20}"
        print(template.format("RSME", "Model", "input_chunk_length", "output_chunk_length")) # header
        for result in results:
            print(template.format(*result))
        #                RSME|               Model|  input_chunk_length| output_chunk_length
        # 0.11735834023309344|           TiDEModel|                  12|                   1
        # 0.12172176559437081|        NLinearModel|                  18|                  14
        # 0.12235252323797512|        DLinearModel|                  20|                   4
        #  0.2232287466509598|            TFTModel|                  20|                   6
    
    
    #test_all_past_future_static_covariate_trained_tfm_against_new_series()


    # ---------------------------------------------------------------------------
    # get_darts_tfm_arguments_optimized_via_backtesting()
    # The above function uses darts_tfm_historical_forecasts()

    # Optimizes the TFM arguments input_chunk_length and output_chunk_length based
    # on your chosen metric (RMSE).  After optimization the model is saved and
    # then it may be used for predictions on new future data.  
    # Does not work with multiple series, but does work with univariate and 
    # multivariate series, and accepts static, past, and future covariates. 

    # This example demonstrates modeling a multivariate series with static, past,
    # and future covariates.  

    # WARNING:  It can take about an hour to run each model:
    #   0 hr 41 min to run NLinearModel
    #   1 hr to run DLinearModel 
    #   1 hr 8 min to run TiDEModel
    #   3 hr 42 min to run TFTModel


    def test_get_darts_tfm_arguments_optimized_via_backtesting():

        from darts.metrics import rmse, rmsle
        from darts.dataprocessing.transformers.scaler import Scaler
        from darts.dataprocessing.transformers import StaticCovariatesTransformer
        from darts import TimeSeries, concatenate
        from darts.utils.statistics import check_seasonality
        import torch

        from time import perf_counter
        from pathlib import Path
        from operator import itemgetter
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np

        import warnings
        warnings.filterwarnings("ignore")
        import logging
        logging.disable(logging.CRITICAL)
        
        # for reproducibility
        # https://vandurajan91.medium.com/random-seeds-and-reproducible-results-in-pytorch-211620301eba
        # https://pytorch.org/docs/stable/notes/randomness.html
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        import random
        np.random.seed(1)

        scaler = Scaler()  # default uses sklearn's MinMaxScaler. 
        transformer = StaticCovariatesTransformer()

        min_max_scale = True

        def get_ts_n_covariates(df=None):

            # Get three Darts univariate TimeSeries from the first 3 columns in df & built a multivariate TimeSeries
            ts1 = TimeSeries.from_series(pd_series=df.iloc[:,0]) 
            ts2 = TimeSeries.from_series(pd_series=df.iloc[:,1]) 
            ts3 = TimeSeries.from_series(pd_series=df.iloc[:,2]) 
            ts = ts1.stack(ts2)
            ts = ts.stack(ts3)
            del ts1, ts2, ts3
            # ts is now a multivariate TimeSeries with 3x components (columns)
            print("\nts: ", ts.time_index.min(), " ..", ts.time_index.max(), "\tlength: ", len(ts.time_index), '\n')  

            # Define static covariates
            # Note: Add any static covariates to series 'ts' BEFORE splitting 'ts' into train, val, test, pred_input..
            static_covariates = None
            static_covariates = pd.DataFrame(data={"cont": [0, 2, 1], "cat": ["a", "c", "b"]})
            if not static_covariates is None:
                # Add the static covariates to the TimeSeries
                ts = ts.with_static_covariates(static_covariates)
                # Encode/transform categorical static covariates to match what was done by get_darts_model_splits() during model training.
                # [[0.0 'a'],[2.0 'c'],[1.0 'b']]   =>  [[0.  0. ],[1.  2. ],[0.5 1. ]]
                ts = transformer.fit_transform(ts)
                print(ts.static_covariates)

            val = None

            # Define any covariates
            past_covariates = None
            future_covariates = None
            val_past_covariates = None
            val_future_covariates = None
            # Create the raw full length series for the covariates from the 4th column of df, one for each in the multivariate series 
            covariates = TimeSeries.from_series(pd_series=df.iloc[:,3]) 
            past_covariates = covariates.stack(covariates) 
            past_covariates = past_covariates.stack(covariates) 
            future_covariates = covariates.stack(covariates) 
            future_covariates = future_covariates.stack(covariates) 
            del covariates
            if not val is None:
                if not past_covariates is None: val_past_covariates = past_covariates.copy()
                if not future_covariates is None: val_future_covariates = future_covariates.copy()

            # Note: min/max scaling done later by darts_tfm_historical_forecasts()

            return ts, val, past_covariates, val_past_covariates, future_covariates, val_future_covariates

        # Define a path to save the data file and the model with the best arguments input_chunk_length & output_chunk_length. 
        # Revise the path below for your system.  Below saves to folder 'data' in the parent of the current working folder.
        # Later in the script, the model filename will be defined using model_save_path with .joinpath() just prior to saving the trained model. 
        model_save_path = Path(Path.cwd().parent).joinpath('data/')
        if not model_save_path.exists(): raise Exception("Folder specified by 'path_file' doesn't exist: "+ str(model_save_path))

        # Generate a data set with 3x signals (multivariate) & a covariate (sine wave).
        # Small size based on max seasonal period of 20: 20 + 20*3 + 20*3 + 20 = 160
        df = sine_gaussian_noise_covariate(length=320, add_trend=False, include_covariate=True, multivariate=True)
        #df.reset_index(inplace=True, drop=True)     # Convert the index from datetime to integer (for testing only)
        
        # Save the generated signal to a Parquet file
        path_file = model_save_path.joinpath('sine_gaussian_noise_multivariate_covariate.parquet')
        df.to_parquet(path=path_file, compression='gzip')
        del df

        # Read the Parquet file to dataframe and assign the first 160 rows to df
        if not path_file.exists(): raise Exception("File doesn't exist: ", path_file)
        df = pd.read_parquet(path=path_file) 
        df = df.iloc[0:160, 0:] # df = df.iloc[0:400, 0:]

        # Get the target series and covariates from df
        ts, val_series, past_covariates, val_past_covariates, future_covariates, val_future_covariates = get_ts_n_covariates(df)

        # How to get the max seasonal period in series ts
        #seasonal_period_max = max(ts_seasonal_periods(ts=ts, sort_ascending=False, unique=False))
        #print("seasonal_period_max:", seasonal_period_max)

        # Define the models to be processed
        from darts.models import (TFTModel,DLinearModel,NLinearModel,TiDEModel)
        # Try each of the Torch Forecasting Models that support past, future, & static covariates and multivariates.
        # Consult the the following table to see what models to choose:  https://github.com/unit8co/darts?tab=readme-ov-file#forecasting-models
        models = {
            #'TFTModel': TFTModel,      # very long solution time, but best result
            'DLinearModel': DLinearModel, 
            'NLinearModel': NLinearModel, 
            'TiDEModel': TiDEModel,
        }

        m_results = []
        for m in models:
            print("\n===========================================================================")
            print("Model: ", m)

            input_chunk_length, output_chunk_length, metric_result, t_elapsed, model = get_darts_tfm_arguments_optimized_via_backtesting(m, models[m], ts=ts, past_covariates=past_covariates, future_covariates=future_covariates, metric=rmse, min_max_scale=True, coarse_only=False)

            if not model_save_path is None:
                # Save the model in order to retain the optimized model arguments.
                # Note: the training series is not saved when the .historical_forecast() method is used.
                # https://unit8co.github.io/darts/userguide/forecasting_overview.html#saving-and-loading-models
                # For TFM, two files are saved:  
                #   the model parameters with the specified extension .pt
                #   the training state with the method added extension of .ckpt
                # WARNING: Do not include “last-” or “best-” in the beginning of the filename.
                # Darts model.save() method doesn't accept a pathlib object, only a string for the path/filename.
                path_file = model_save_path.joinpath(m + ".pt")
                print("Saving model " + m + " to: " + str(path_file))
                model.save(str(path_file))

            print(metric_result, m, input_chunk_length, output_chunk_length, sec_to_dhms(t_elapsed))
            m_results.append([metric_result, m, input_chunk_length, output_chunk_length, sec_to_dhms(t_elapsed)])

        print("\nResults across all models:")
        for result in m_results:
            print(*result)
        print("\n")


        # Run a prediction for the model with the best fit during input_chunk_length & output_chunk_length optimization. 

        # Read the saved model
        m = m_results[0][1]
        path_file = model_save_path.joinpath(m + ".pt")
        if not path_file.exists(): raise Exception("File doesn't exist: ", path_file)
        model = models[m].load(str(path_file))
        print("Optimized model arguments:")
        print("\tinput_chunk_length:", model.input_chunk_length)
        print("\toutput_chunk_length:", model.output_chunk_length)

        if model.training_series is None:
            # Train the model on df

            # Read the Parquet file to dataframe and assign the first 160 rows to df
            path_file = model_save_path.joinpath('sine_gaussian_noise_multivariate_covariate.parquet')
            if not path_file.exists(): raise Exception("File doesn't exist: ", path_file)
            df = pd.read_parquet(path=path_file) 
            df = df.iloc[0:160, 0:]     #df = df.iloc[0:400, 0:]

            ts, val_series, past_covariates, val_past_covariates, future_covariates, val_future_covariates = get_ts_n_covariates(df)
            if m == 'TCNModel':
                future_covariates = None
            
            # Trim the target series (for training & prediction) to the maximum length (limited by past/future coveriates & input_chunk_length/output_chunk_length)
            if past_covariates is None and future_covariates is None:
                series = ts
            else:
                # Series ts must be smaller than future_covariates & past covariates.  
                #print("Min series length for past & future covariates: ", len(ts)-input_chunk_length-output_chunk_length+1)
                series = ts[model.input_chunk_length+1:len(ts)-model.output_chunk_length]

            if min_max_scale == True: 
                series = scaler.fit_transform(series)
                if not past_covariates is None: past_covariates = scaler.transform(past_covariates)
                if not val_past_covariates is None: val_past_covariates = scaler.transform(val_past_covariates)
                if not future_covariates is None: future_covariates = scaler.transform(future_covariates)
                if not val_future_covariates is None: val_future_covariates = scaler.transform(val_future_covariates)

            model.fit(series=series, past_covariates=past_covariates, future_covariates=future_covariates, val_series=val_series, val_past_covariates=val_past_covariates, val_future_covariates=val_future_covariates)

            # save the trained model
            # For TFM, two files are saved:  
            #   the model parameters with the specified extension .pt
            #   the training state with the method added extension of .ckpt
            # WARNING: Do not include “last-” or “best-” in the beginning of the filename.
            path_file = model_save_path.joinpath(m + "_trained.pt")
            print("Saving trained model " + m + " to: " + str(path_file))
            # Darts model.save() method doesn't accept a pathlib object, only a string for the path/filename.
            model.save(str(path_file))

        print("\nmodel.training_series: ", model.training_series.time_index.min(), " ..", model.training_series.time_index.max(), "\tlength: ", len(model.training_series.time_index))  

        # Read the Parquet file to dataframe and assign the last 159 rows to df
        path_file = model_save_path.joinpath('sine_gaussian_noise_multivariate_covariate.parquet')
        if not path_file.exists(): raise Exception("File doesn't exist: ", path_file)
        df = pd.read_parquet(path=path_file) 
        df = df.iloc[160:, 0:]      # df = df.iloc[400:, 0:]

        # Get the target series and covariates from df
        ts, val_series, past_covariates, val_past_covariates, future_covariates, val_future_covariates = get_ts_n_covariates(df)

        if min_max_scale == True: 
            ts = scaler.fit_transform(ts)
            if not past_covariates is None: past_covariates = scaler.transform(past_covariates)
            if not val_past_covariates is None: val_past_covariates = scaler.transform(val_past_covariates)
            if not future_covariates is None: future_covariates = scaler.transform(future_covariates)
            if not val_future_covariates is None: val_future_covariates = scaler.transform(val_future_covariates)

        # Create a series pred_input from ts for input to .predict(series=pred_input) with a length > input_chunk_length
        # start at row 20 and end with a length of 20*3 rows
        pred_input = ts[20:20+20*3]         
        # How to slice by specific index date values:                            
        #pred_input = ts.slice(pd.Timestamp(year=2006, month=6, day=1), pd.Timestamp(year=2006, month=9, day=8))
        print("\npred_input: ", pred_input.time_index.min(), " ..", pred_input.time_index.max(), "\tlength: ", len(pred_input.time_index))  

        # Define the the number of steps (series rows) or n beyond the end of pred_input for the prediction.
        pred_steps = 20*3

        # Below will optionally slice the covariates to the minimim start/end required for the model .predict() method.  
        # This is useful for verifying the covariates have the correct time span relative to the target series.  
        #cv_past, cv_future, cv_static = slice_tfm_covariates_for_trained_model(model, ts, pred_input, pred_steps, past_covariates, future_covariates, min_max_scale=False, verbose=True)

        # Execute a prediction
        pred = model.predict(n=pred_steps, series=pred_input, past_covariates=past_covariates, future_covariates=future_covariates)  # defaults to series=train when pred_input==None
        print("\npred: ", pred.time_index.min(), " ..", pred.time_index.max(), "\tlength: ", len(pred.time_index))  

        # Calculate the root mean square error (RMSE)
        metric_rmse = rmse(ts.slice_intersect(pred), pred)
        print(m + " RMSE: ", round(metric_rmse,5))   
        
        # Plot
        title = m + " RMSE: " + str(round(metric_rmse,5))
        plt_model_trained(ts, past_covariates, future_covariates, pred_input, pred, title)


    test_get_darts_tfm_arguments_optimized_via_backtesting()


    # ---------------------------------------------------------------------------
    # Testing of get_darts_tfm_arguments_optimized_via_backtesting() against 
    # various Darts datasets.

    def test_get_darts_tfm_arguments_optimized_via_backtesting_with_darts_datasets():

        from darts import TimeSeries
        from darts.dataprocessing.transformers.scaler import Scaler
        from darts.dataprocessing.transformers import StaticCovariatesTransformer
        from darts.metrics import rmse
        import torch

        from pathlib import Path
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        import warnings
        warnings.filterwarnings("ignore")
        import logging
        logging.disable(logging.CRITICAL)

        #from savvy_time_series import get_idx_datetime_attr, index_has_gaps, df_has_nan_or_nat, sec_to_dhms, reset_matplotlib, get_darts_tfm_arguments_optimized_via_backtesting, plt_darts_ts_stacked, ts_seasonal_periods

        # for reproducibility
        torch.manual_seed(1)
        np.random.seed(1)

        scaler = Scaler()  # default uses sklearn's MinMaxScaler. 
        transformer = StaticCovariatesTransformer()

        # Set min_max_scale = True to min/max scale the data to range of 0.0 to +1.0
        min_max_scale = True

        # Define a path to save the data file and the model with the best arguments input_chunk_length & output_chunk_length. 
        # Revise the path below for your system.  Below saves to folder 'data' in the parent of the current working folder.
        # Later in the script, the model filename will be defined using model_save_path with .joinpath() just prior to saving the trained model. 
        model_save_path = Path(Path.cwd().parent).joinpath('data/')
        if not model_save_path.exists(): raise Exception("Folder specified by 'model_save_path' doesn't exist: "+ str(model_save_path))

        from darts.datasets import (
            AirPassengersDataset,
            AusBeerDataset,
            GasRateCO2Dataset,
            HeartRateDataset,
            IceCreamHeaterDataset,
            #MonthlyMilkIncompleteDataset,
            MonthlyMilkDataset,
            MonthlyMilkDataset,
            SunspotsDataset,
            WeatherDataset, 
            AustralianTourismDataset,
            EnergyDataset,
            TaylorDataset, 
            TemperatureDataset, 
            USGasolineDataset, 
            WineDataset,
            WoolyDataset,
            EnergyDataset,
            AustralianTourismDataset,
            ETTh1Dataset,
            ETTh2Dataset,
            ETTm1Dataset,
            ETTm2Dataset,
            ElectricityDataset,
            ILINetDataset,
            ExchangeRateDataset,
            TrafficDataset,
            #UberTLCDataset,
        )

        datasets = [
            AirPassengersDataset,          # 1 cols x 144 rows     seasonal 12 with non-linear increasing trend     
            #AusBeerDataset,                # 1 cols x 211 rows     SeasonalityMode.MULTIPLICATIVE 4, ModelMode.MULTIPLICATIVE, non-linear   
            #IceCreamHeaterDataset,         # 2 cols x 198 rows     SeasonalityMode.MULTIPLICATIVE 12, with non-linear increasing trend ModelMode.MULTIPLICATIVE   
            #MonthlyMilkDataset,            # 1 cols x 168 rows     SeasonalityMode.MULTIPLICATIVE 12, increasing linear trend ModelMode.MULTIPLICATIVE   
            #MonthlyMilkIncompleteDataset,  # 1 cols x 168 rows        *** NaN & index gaps ***  datetime index MS | increasing trend | SeasonalityMode.MULTIPLICATIVE with a period of 12
            #GasRateCO2Dataset,             # 2 cols x 296 rows,    SeasonalityMode.ADDITIVE 23,  ModelMode.NONE, no trend with data about mean of 0    
            #HeartRateDataset,              # 1 cols x 1800 rows,   not-seasonal, random about mean near 90      ***
            #SunspotsDataset,                # 1 cols x 2820 rows,   not-seasonal random about mean near 40 with + bias    
            #WeatherDataset,                # 21 cols x 52704 rows!!!  not-seasonal    *** Has NaN / NaT values ***
            #AustralianTourismDataset,      # 96 cols x 36 rows     
            #TaylorDataset,                 # 1 cols x 4032 rows     index range/int type, trend: linear / additive, SeasonalityMode.NONE
            #TemperatureDataset,            # 1 cols x 3652 rows      NaN/NaT values | datetime index | trend: | SeasonalityMode.ADDITIVE with period 5 
            #USGasolineDataset,             # 1 cols x 1578 rows     datetime index | trend: ModelMode.MULTIPLICATIVE | SeasonalityMode.NONE
            #WineDataset,                   # 1 cols x 176 rows      index datetime | trend: ModelMode.MULTIPLICATIVE | SeasonalityMode.MULTIPLICATIVE with period of 4 and 12
            #WoolyDataset,                  #1 cols x 119 rows       seasonal 4/30
            #EnergyDataset,                 # 28 cols x 35064 rows   *** NaN/NaT values that cannot be automatically resolved ***
            #AustralianTourismDataset,      # 96 cols x 36 rows      range index | no trend | SeasonalityMode.MULTIPLICATIVE with period of 4
            #ETTh1Dataset,                  # 7 cols x 17420 rows    datetime index | trend: no ??   | SeasonalityMode.ADDITIVE period of 15
            #ETTh2Dataset,                  # 7 cols x 17420 rows    datetime index | no trend | SeasonalityMode.ADDITIVE with period of 15
            #ETTm1Dataset,                  # 7 cols x 69680 rows    *** datetime index 15T | no trend | SeasonalityMode.NONE | RandomForest MemoryError: could not allocate 25572671488 bytes (25.6 GB)
            #ETTm2Dataset,                  # 7 cols x 69680 rows   
            #ElectricityDataset,            # 370 cols x 140256 rows     datetime index 15T | 
            #ExchangeRateDataset,           # 8 cols x 7588 rows     range index | no trend | SeasonalityMode.NONE 
            ##ILINetDataset,                 # 11 cols x 1305 rows    datetime index W-SUN | *** columns have NaN and index has gaps ***
            #TrafficDataset,                # 862 cols x 17544 rows      datetime index H | 
            #UberTLCDataset,                # *** does not load ***
        ]
        # https://unit8co.github.io/darts/generated_api/darts.datasets.html

        for dataset in datasets:
            ts = dataset().load()
            print("\n\n# ---------------------------------------------------------------------------")
            print("Dataset: ", dataset.__name__, "  " + str(ts.width) + " cols x " + str(len(ts)) + " rows")

            # Get basic information about the dataset and check for issues..
            if ts.has_datetime_index:
                print("Series has datetime index with frequency of '" + str(ts.freq) + "' or '" + str(ts.freq_str) + "'")
                # Below tests custom function get_idx_datetime_attr() to be sure the value of ts.freq_str is encoded in the function.
                # If an error is raised, go to get_idx_datetime_attr() and edit it.  
                print("get_idx_datetime_attr(ts.freq_str):", get_idx_datetime_attr(ts.freq_str))
            else:
                print("Series has range index (int) with unit '" + str(ts.time_dim) + "' and definition of: " + str(ts.time_index))

            # Check for gaps in the index caused by NaN/NaT
            if index_has_gaps(ts.pd_dataframe()): raise Exception("ts has gaps in the index likely caused by NaN/NaT")

            # Check the series components/columns for NaN/NaT values and attempt to resolve if found.
            if df_has_nan_or_nat(ts.pd_dataframe(), results_by_col=False, verbose=False): 
                df_has_nan_or_nat(ts.pd_dataframe(), results_by_col=False, verbose=True)
                len_ts = len(ts)
                err = False
                try:
                    ts = ts.longest_contiguous_slice()
                except Exception as e:
                    print("\t", "ERROR: " + repr(e))
                    err = True
                if not err:
                    print("WARNING: extracted longest continguous slice without NaN/NaT, series was " + str(len_ts) + " rows, now " + str(len(ts)), "\n")
                else:
                    # Alternative method of dealing with NaN/NaT
                    print("WARNING:  Dropping these rows with NaN/NaT:")
                    df = ts.pd_dataframe()
                    # print the rows that have NaN/NaT
                    print(df[df.isnull().any(axis=1)])
                    # drop the rows with NaN/NaT
                    df.dropna(inplace=True)
                    # Create ts from df, but fix missing dates created by .dropna()
                    ts = TimeSeries.from_dataframe(df, fill_missing_dates=True)
                if index_has_gaps(ts.pd_dataframe()): raise Exception("The series has one or more columns with NaN / NaT values that cannot be easily reconciled")

            past_covariates = None
            future_covariates = None
            val_past_covariates = None
            val_future_covariates = None

            if min_max_scale == True: 
                # min/max scale the data to a range of 0.0 to +1.0
                ts = scaler.fit_transform(ts)
                if not past_covariates is None: past_covariates = scaler.transform(past_covariates)
                if not val_past_covariates is None: val_past_covariates = scaler.transform(val_past_covariates)
                if not future_covariates is None: future_covariates = scaler.transform(future_covariates)
                if not val_future_covariates is None: val_future_covariates = scaler.transform(val_future_covariates)

            # Define the models to be processed
            # Try each of the Torch Forecasting Models that don't require past or future covariates.
            # Consult the the following table to see what models to choose:  https://github.com/unit8co/darts?tab=readme-ov-file#forecasting-models
            from darts.models import (DLinearModel,NLinearModel,TiDEModel,TCNModel)
            models = {
                #'DLinearModel': DLinearModel, 
                #'NLinearModel': NLinearModel, 
                'TiDEModel': TiDEModel,
                #'TCNModel': TCNModel,
            }

            m_results = []
            for m in models:
                print("\n===========================================================================")
                print("Model: ", m)

                input_chunk_length, output_chunk_length, metric_result, t_elapsed, model = get_darts_tfm_arguments_optimized_via_backtesting(m, models[m], ts=ts, past_covariates=past_covariates, future_covariates=future_covariates, metric=rmse, min_max_scale=False, coarse_only=False)

                if not model_save_path is None:
                    # Save the model in order to retain the optimized model arguments.
                    # Note: the training series is not saved when the .historical_forecast() method is used.
                    # https://unit8co.github.io/darts/userguide/forecasting_overview.html#saving-and-loading-models
                    # For TFM, two files are saved:  
                    #   the model parameters with the specified extension .pt
                    #   the training state with the method added extension of .ckpt
                    # WARNING: Do not include “last-” or “best-” in the beginning of the filename.
                    # Darts model.save() method doesn't accept a pathlib object, only a string for the path/filename.
                    path_file = model_save_path.joinpath(dataset.__name__ + "_" + m + ".pt")
                    print("Saving model " + m + " to: " + str(path_file))
                    model.save(str(path_file))

                print(metric_result, m, input_chunk_length, output_chunk_length, sec_to_dhms(t_elapsed))
                m_results.append([metric_result, m, input_chunk_length, output_chunk_length, sec_to_dhms(t_elapsed)])

            print("\nResults across all models:")
            for result in m_results:
                print(*result)

            # Run a prediction using the best saved model and the optimized input_chunk_length & output_chunk_length

            # Read the saved model
            m = m_results[0][1]
            print("Reading saved model " + m + " with optimized input_chunk_length and output_chunk_length..")
            path_file = model_save_path.joinpath(dataset.__name__ + "_" + m + ".pt")
            if not path_file.exists(): raise Exception("File doesn't exist: ", path_file)
            print(path_file)
            model = models[m].load(str(path_file))
            print("Optimized model arguments:")
            print("\tinput_chunk_length:", model.input_chunk_length)
            print("\toutput_chunk_length:", model.output_chunk_length)

            # Split the data for training, fit, predict
            # When series is seasonal, input_chunk_length typically equals the seasonality period, and *3 allows the prediction to be at least over 3 seasonal periods.
            """
            pred_steps = int(model.input_chunk_length*3)    
            if past_covariates is None and future_covariates is None:
                series = ts[:len(ts)-pred_steps*2]
            else:
                # series must be smaller than future_covariates & past covariates.  
                series = ts[model.input_chunk_length+1:len(ts)-4-pred_steps*2]
            # Create pred_input so it starts at the next row / time increment after the end of series, with a length equal to pred_steps
            pred_input = ts.slice(series.time_index.max(), ts.time_index.max())
            pred_input = pred_input[1:pred_steps]
            val_series = None
            print("\nData Splits From ts For Fit & Predict:")
            print("ts: ", ts.time_index.min(), " ..", ts.time_index.max(), "\tlength: ", len(ts.time_index))  
            print("series: ", series.time_index.min(), " ..", series.time_index.max(), "\tlength: ", len(series.time_index))  
            if not pred_input is None: print("pred_input: ", pred_input.time_index.min(), " ..", pred_input.time_index.max(), "\tlength: ", len(pred_input.time_index))  
            print("pred_steps:", pred_steps)
            if len(series)+len(pred_input)+pred_steps > len(ts): raise Exception("ERROR: splits of series, pred_input, and pred_steps exceed the length of the source series ts")
            print("")
            """

            pred_steps = int(model.input_chunk_length*3)    
            if past_covariates is None and future_covariates is None:
                series = ts[:len(ts)-pred_steps]
            else:
                # series must be smaller than future_covariates & past covariates.  
                series = ts[model.input_chunk_length+1:len(ts)-4-pred_steps]
            pred_input = None
            val_series = None
            print("\nData Splits From ts For Fit & Predict:")
            print("ts: ", ts.time_index.min(), " ..", ts.time_index.max(), "\tlength: ", len(ts.time_index))  
            print("series: ", series.time_index.min(), " ..", series.time_index.max(), "\tlength: ", len(series.time_index))  
            print("pred_steps:", pred_steps)
            if len(series)+pred_steps > len(ts): raise Exception("ERROR: The length of series + pred_steps exceeds the length of the source series ts")
            print("")

            if model.training_series is None:
                # Train the model 

                model.fit(series=series, past_covariates=past_covariates, future_covariates=future_covariates, val_series=val_series, val_past_covariates=val_past_covariates, val_future_covariates=val_future_covariates)

                # save the trained model
                # For TFM, two files are saved:  
                #   the model parameters with the specified extension .pt
                #   the training state with the method added extension of .ckpt
                # WARNING: Do not include “last-” or “best-” in the beginning of the filename.
                path_file = model_save_path.joinpath(dataset.__name__ + "_" + m + "_trained.pt")
                print("Saving trained model " + m + " to: " + str(path_file))
                # Darts model.save() method doesn't accept a pathlib object, only a string for the path/filename.
                model.save(str(path_file))

            print("\nmodel.training_series: ", model.training_series.time_index.min(), " ..", model.training_series.time_index.max(), "\tlength: ", len(model.training_series.time_index))  

            if not pred_input is None: 
                pred = model.predict(n=pred_steps, series=pred_input, past_covariates=past_covariates, future_covariates=future_covariates)  # defaults to series=train when pred_input==None
            else:
                pred = model.predict(n=pred_steps, series=series, past_covariates=past_covariates, future_covariates=future_covariates)  # defaults to series=train when pred_input==None
            print("\npred: ", pred.time_index.min(), " ..", pred.time_index.max(), "\tlength: ", len(pred.time_index))  

            # Calculate the root mean square error (RMSE)
            metric_rmse = rmse(ts.slice_intersect(pred), pred)
            print(m + " RMSE: ", round(metric_rmse,5))   

            if min_max_scale:
                # Scale all series back to the original scale
                ts = scaler.inverse_transform(ts)
                series = scaler.inverse_transform(series)
                if not pred_input is None: pred_input = scaler.inverse_transform(pred_input)
                pred = scaler.inverse_transform(pred)

            # Plot
            title = dataset.__name__ + " Full Scale Results for " + m + " RMSE: " + str(round(metric_rmse,5))
            reset_matplotlib()
            ts.slice_intersect(pred).plot(label='ground truth', linewidth=3.0, linestyle='--', color='tab:gray')
            if not pred_input is None: pred_input.plot(label='pred_input')
            series.plot(label='series')
            pred.plot(label='pred')
            plt.title(title)
            plt.show()

    #test_get_darts_tfm_arguments_optimized_via_backtesting_with_darts_datasets()


    # ---------------------------------------------------------------------------

