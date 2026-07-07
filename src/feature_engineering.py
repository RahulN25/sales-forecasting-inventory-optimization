import pandas as pd
import numpy as np

def create_date_features(df):
    """
    Extracts date-related components from the 'Date' column.
    Assumes df['Date'] is already converted to datetime.
    """
    df = df.copy()
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['Quarter'] = df['Date'].dt.quarter
    df['Is_Q4'] = (df['Quarter'] == 4).astype(int)
    # Holiday months are Feb (Super Bowl), Sep (Labor Day), Nov (Thanksgiving), Dec (Christmas)
    df['Is_Holiday_Month'] = df['Month'].isin([2, 9, 11, 12]).astype(int)
    return df


def create_holiday_details(df):
    """
    Creates specific binary indicators for major Walmart holidays.
    Dates are mapped from historical Walmart holiday periods:
    - Super Bowl: Feb 12, 2010; Feb 11, 2011; Feb 10, 2012; Feb 8, 2013
    - Labor Day: Sep 10, 2010; Sep 9, 2011; Sep 7, 2012; Sep 6, 2013
    - Thanksgiving: Nov 26, 2010; Nov 25, 2011; Nov 23, 2012; Nov 29, 2013
    - Christmas: Dec 31, 2010; Dec 30, 2011; Dec 28, 2012; Dec 27, 2013
    """
    df = df.copy()
    
    super_bowl_dates = pd.to_datetime(['2010-02-12', '2011-02-11', '2012-02-10', '2013-02-08'])
    labor_day_dates = pd.to_datetime(['2010-09-10', '2011-09-09', '2012-09-07', '2013-09-06'])
    thanksgiving_dates = pd.to_datetime(['2010-11-26', '2011-11-25', '2012-11-23', '2013-11-29'])
    christmas_dates = pd.to_datetime(['2010-12-31', '2011-12-30', '2012-12-28', '2013-12-27'])
    
    df['Is_SuperBowl'] = df['Date'].isin(super_bowl_dates).astype(int)
    df['Is_LaborDay'] = df['Date'].isin(labor_day_dates).astype(int)
    df['Is_Thanksgiving'] = df['Date'].isin(thanksgiving_dates).astype(int)
    df['Is_Christmas'] = df['Date'].isin(christmas_dates).astype(int)
    
    return df


