# Q_2.1


```python
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from scipy.optimize import minimize
import matplotlib.pyplot as plt

```


```python


df = pd.read_csv('synthetic_control_germany_robust.csv')

print("Columns:", df.columns.tolist())
print("\n First rows:")
print(df.head())
print("\n Country:", df['entity_id'].unique())
print("\n Data information:")
print(df.info())

imputer = KNNImputer(n_neighbors=5)
numeric_cols = ['indicator_score', 'metric_inflation', 'trade', 'metric_education', 'metric_industry_share']
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

id_column = 'entity_id'
treated_unit = 'West Germany'
donor_pool = [entity for entity in df[id_column].unique() if entity != treated_unit]
print("\nDonor pool:", donor_pool)

pre_treatment = df[df['year_recorded'] < 1990]


treated_data = pre_treatment[pre_treatment[id_column] == treated_unit]
print("\n West germany data in pre_treatment:")
print(treated_data.head())

covariates = ['metric_inflation', 'trade', 'metric_education', 'metric_industry_share']
X1 = pre_treatment[pre_treatment[id_column] == treated_unit][covariates].mean().values
X0 = (pre_treatment[pre_treatment[id_column].isin(donor_pool)][covariates + [id_column]]
      .groupby(id_column)[covariates].mean().values)

Y1 = pre_treatment[pre_treatment[id_column] == treated_unit]['indicator_score'].values
Y0 = (pre_treatment[pre_treatment[id_column].isin(donor_pool)]
      [['year_recorded', id_column, 'indicator_score']]
      .pivot(index='year_recorded', columns=id_column, values='indicator_score').values)

print("\n Len Y1 (GDP West germany before intervention):", len(Y1))
print("Shape Y0 (GDP Countries donor):", Y0.shape)

def objective_function(weights):
    if len(Y1) == 0 or Y0.size == 0:
        print("Y0 or Y1 is empty!")
        return np.inf
    return np.sqrt(np.mean((Y1 - Y0 @ weights) ** 2))

constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
bounds = [(0, 1)] * len(donor_pool)
initial_weights = np.ones(len(donor_pool)) / len(donor_pool)

result = minimize(objective_function, initial_weights, bounds=bounds, constraints=constraints)
weights = result.x

synthetic_gdp = (df[df[id_column].isin(donor_pool)][['year_recorded', id_column, 'indicator_score']]
                 .pivot(index='year_recorded', columns=id_column, values='indicator_score').values @ weights)

synthetic_df = pd.DataFrame({'year': df['year_recorded'].unique(), 'synthetic_gdp': synthetic_gdp})
actual_df = df[df[id_column] == treated_unit][['year_recorded', 'indicator_score']].rename(
    columns={'indicator_score': 'actual_gdp'})
result_df = actual_df.merge(synthetic_df, left_on='year_recorded', right_on='year')

print("\n  donor pool Weights:")
for entity, weight in zip(donor_pool, weights):
    print(f"{entity}: {weight:.3f}")
print("\n Synthetic Control Result (First rows):")
print(result_df.head())
```

    Columns: ['entity_id', 'year_recorded', 'indicator_score', 'group_tag', 'metric_inflation', 'trade', 'metric_education', 'metric_industry_share']
    
     First rows:
      entity_id  year_recorded  indicator_score    group_tag  metric_inflation  \
    0       USA           1960             2879  GermanyCase               NaN   
    1       USA           1961             2929  GermanyCase          1.075182   
    2       USA           1962             3103  GermanyCase          1.116071   
    3       USA           1963             3227  GermanyCase          1.214128   
    4       USA           1964             3420  GermanyCase          1.308615   
    
          trade  metric_education  metric_industry_share  
    0  9.693181         43.799999                    NaN  
    1  9.444655               NaN                    NaN  
    2  9.429324               NaN                    NaN  
    3  9.470706               NaN                    NaN  
    4  9.725879               NaN                    NaN  
    
     Country: ['USA' 'UK' 'Austria' 'Belgium' 'Denmark' 'France' 'West Germany' 'Italy'
     'Netherlands' 'Norway' 'Switzerland' 'Japan' 'Greece' 'Portugal' 'Spain'
     'Australia' 'New Zealand']
    
     Data information:
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 748 entries, 0 to 747
    Data columns (total 8 columns):
     #   Column                 Non-Null Count  Dtype  
    ---  ------                 --------------  -----  
     0   entity_id              748 non-null    object 
     1   year_recorded          748 non-null    int64  
     2   indicator_score        748 non-null    int64  
     3   group_tag              748 non-null    object 
     4   metric_inflation       727 non-null    float64
     5   trade                  646 non-null    float64
     6   metric_education       151 non-null    float64
     7   metric_industry_share  541 non-null    float64
    dtypes: float64(4), int64(2), object(2)
    memory usage: 46.9+ KB
    None
    
    Donor pool: ['USA', 'UK', 'Austria', 'Belgium', 'Denmark', 'France', 'Italy', 'Netherlands', 'Norway', 'Switzerland', 'Japan', 'Greece', 'Portugal', 'Spain', 'Australia', 'New Zealand']
    
     West germany data in pre_treatment:
            entity_id  year_recorded  indicator_score    group_tag  \
    264  West Germany           1960           2284.0  GermanyCase   
    265  West Germany           1961           2388.0  GermanyCase   
    266  West Germany           1962           2527.0  GermanyCase   
    267  West Germany           1963           2610.0  GermanyCase   
    268  West Germany           1964           2806.0  GermanyCase   
    
         metric_inflation      trade  metric_education  metric_industry_share  
    264          3.992297  35.459682             61.00              44.987244  
    265          2.515723  33.836784             39.00              38.540255  
    266          2.760736  33.466377             29.88              38.540255  
    267          2.985075  34.160629             25.34              38.736319  
    268          2.318840  34.566139             27.72              37.100947  
    
     Len Y1 (GDP West germany before intervention): 30
    Shape Y0 (GDP Countries donor): (30, 16)
    
      donor pool Weights:
    USA: 0.000
    UK: 0.323
    Austria: 0.000
    Belgium: 0.000
    Denmark: 0.039
    France: 0.099
    Italy: 0.061
    Netherlands: 0.000
    Norway: 0.000
    Switzerland: 0.000
    Japan: 0.028
    Greece: 0.000
    Portugal: 0.000
    Spain: 0.108
    Australia: 0.000
    New Zealand: 0.343
    
     Synthetic Control Result (First rows):
       year_recorded  actual_gdp  year  synthetic_gdp
    0           1960      2284.0  1960    2220.382726
    1           1961      2388.0  1961    2343.901315
    2           1962      2527.0  1962    2464.443074
    3           1963      2610.0  1963    2582.937583
    4           1964      2806.0  1964    2747.062973


# Q_2.2

**bold text**


```python
plt.figure(figsize=(10, 6))
plt.plot(result_df['year_recorded'], result_df['actual_gdp'], label='Actual GDP (West Germany)', color='blue')
plt.plot(result_df['year_recorded'], result_df['synthetic_gdp'], label='Synthetic GDP', linestyle='--', color='red')
plt.axvline(x=1990, color='black', linestyle=':', label='Reunification (1990)')
plt.title('Actual vs Synthetic GDP for West Germany (1960-2003)')
plt.xlabel('Year')
plt.ylabel('GDP')
plt.legend()
plt.grid()
plt.show()


result_df['pointwise_gap'] = result_df['actual_gdp'] - result_df['synthetic_gdp']
plt.figure(figsize=(10, 6))
plt.plot(result_df['year_recorded'], result_df['pointwise_gap'], label='Pointwise Gap', color='green')
plt.axvline(x=1990, color='black', linestyle=':', label='Reunification (1990)')
plt.axhline(y=0, color='gray', linestyle='--')
plt.title('Pointwise Gap (Actual - Synthetic GDP) for West Germany')
plt.xlabel('Year')
plt.ylabel('Gap (Actual - Synthetic)')
plt.legend()
plt.grid()
plt.show()


post_treatment = result_df[result_df['year_recorded'] >= 1990].copy()
post_treatment['cumulative_gap'] = post_treatment['pointwise_gap'].cumsum()
plt.figure(figsize=(10, 6))
plt.plot(post_treatment['year_recorded'], post_treatment['cumulative_gap'], label='Cumulative Gap', color='purple')
plt.title('Cumulative Gap After Reunification (1990-2003)')
plt.xlabel('Year')
plt.ylabel('Cumulative Gap')
plt.legend()
plt.grid()
plt.show()


print("\nPointwise Gap In first years after intervention:")
print(post_treatment[['year_recorded', 'pointwise_gap']].head())
print("\nCumulative Gap In last years:")
print(post_treatment[['year_recorded', 'cumulative_gap']].tail())
```


    
    


    
    


    
    


    
    Pointwise Gap In first years after intervention:
        year_recorded  pointwise_gap
    30           1990     326.536317
    31           1991     746.451220
    32           1992     558.712785
    33           1993    -194.273523
    34           1994    -653.883670
    
    Cumulative Gap In last years:
        year_recorded  cumulative_gap
    39           1999    -6430.199591
    40           2000    -9013.597944
    41           2001   -11771.197463
    42           2002   -14717.436813
    43           2003   -18163.662562


# Part d


```python


placebo_year = 1982

id_column = 'entity_id'
treated_unit = 'West Germany'
donor_pool = [entity for entity in df[id_column].unique() if entity != treated_unit]

pre_treatment = df[df['year_recorded'] < placebo_year]

covariates = ['metric_inflation', 'trade', 'metric_education', 'metric_industry_share']
X1 = pre_treatment[pre_treatment[id_column] == treated_unit][covariates].mean().values
X0 = (pre_treatment[pre_treatment[id_column].isin(donor_pool)][covariates + [id_column]]
      .groupby(id_column)[covariates].mean().values)

Y1 = pre_treatment[pre_treatment[id_column] == treated_unit]['indicator_score'].values
Y0 = (pre_treatment[pre_treatment[id_column].isin(donor_pool)]
      [['year_recorded', id_column, 'indicator_score']]
      .pivot(index='year_recorded', columns=id_column, values='indicator_score').values)

def objective_function(weights):
    if len(Y1) == 0 or Y0.size == 0:
        return np.inf
    return np.sqrt(np.mean((Y1 - Y0 @ weights) ** 2))

constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
bounds = [(0, 1)] * len(donor_pool)
initial_weights = np.ones(len(donor_pool)) / len(donor_pool)
result = minimize(objective_function, initial_weights, bounds=bounds, constraints=constraints)
weights = result.x

pivot_table = df[df[id_column].isin(donor_pool)][['year_recorded', id_column, 'indicator_score']]
synthetic_gdp = (pivot_table.pivot(index='year_recorded', columns=id_column, values='indicator_score').values @ weights)

synthetic_df = pd.DataFrame({'year': df['year_recorded'].unique(), 'synthetic_gdp': synthetic_gdp})
actual_df = df[df[id_column] == treated_unit][['year_recorded', 'indicator_score']].rename(
    columns={'indicator_score': 'actual_gdp'})
result_df = actual_df.merge(synthetic_df, left_on='year_recorded', right_on='year')

result_df['gap'] = result_df['actual_gdp'] - result_df['synthetic_gdp']

plt.figure(figsize=(10, 5))
plt.plot(result_df['year_recorded'], result_df['gap'], label='Gap (West Germany - Synthetic)', color='b')
plt.axvline(placebo_year, color='r', linestyle='--', label='Placebo Treatment Year (1982)')
plt.xlabel('Year')
plt.ylabel('Gap in GDP')
plt.title('In-Time Placebo Test: Gap after 1982')
plt.legend()
plt.grid()
plt.show()

```


    
    


# Part e


```python


id_column = 'entity_id'
treated_unit = 'West Germany'
donor_pool = [entity for entity in df[id_column].unique() if entity != treated_unit]
treatment_year = 1990
years = df['year_recorded'].unique()

def compute_scm_gap(treated, donor_pool):
    pre_treatment = df[df['year_recorded'] < treatment_year]
    post_treatment = df[df['year_recorded'] >= treatment_year]
    covariates = ['metric_inflation', 'trade', 'metric_education', 'metric_industry_share']

    Y1 = pre_treatment[pre_treatment[id_column] == treated]['indicator_score'].values
    Y0 = (pre_treatment[pre_treatment[id_column].isin(donor_pool)]
        [['year_recorded', id_column, 'indicator_score']]
        .pivot(index='year_recorded', columns=id_column, values='indicator_score').values)

    def objective_function(weights):
        return np.sqrt(np.mean((Y1 - Y0 @ weights) ** 2))

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1)] * len(donor_pool)
    initial_weights = np.ones(len(donor_pool)) / len(donor_pool)
    result = minimize(objective_function, initial_weights, bounds=bounds, constraints=constraints)
    weights = result.x

    pivot_table = df[df[id_column].isin(donor_pool)][['year_recorded', id_column, 'indicator_score']]
    synthetic_gdp = (pivot_table.pivot(index='year_recorded', columns=id_column, values='indicator_score').values @ weights)

    actual_df = df[df[id_column] == treated][['year_recorded', 'indicator_score']].rename(
        columns={'indicator_score': 'actual_gdp'})
    synthetic_df = pd.DataFrame({'year_recorded': years, 'synthetic_gdp': synthetic_gdp})
    result_df = actual_df.merge(synthetic_df, on='year_recorded')
    result_df['gap'] = result_df['actual_gdp'] - result_df['synthetic_gdp']

    pre_gap = result_df[result_df['year_recorded'] < treatment_year]['gap'].values
    pre_rmspe = np.sqrt(np.mean(pre_gap ** 2))
    return result_df['year_recorded'], result_df['gap'], pre_rmspe

years, gap_wg, rmspe_wg = compute_scm_gap(treated_unit, donor_pool)

all_gaps = []
all_rmspe = []
names = []
for country in donor_pool:
    donor_pool2 = [x for x in donor_pool if x != country] + [treated_unit]  # بقیه کشورها + آلمان غربی
    years_i, gap_i, rmspe_i = compute_scm_gap(country, donor_pool2)
    all_gaps.append(gap_i)
    all_rmspe.append(rmspe_i)
    names.append(country)

rmspe_threshold = 20 * rmspe_wg
filtered_gaps = [gap for gap, rmspe in zip(all_gaps, all_rmspe) if rmspe < rmspe_threshold]
filtered_names = [name for name, rmspe in zip(names, all_rmspe) if rmspe < rmspe_threshold]

plt.figure(figsize=(12, 6))
for gap, name in zip(filtered_gaps, filtered_names):
    plt.plot(years, gap, color='gray', alpha=0.5, label='_nolegend_')
plt.plot(years, gap_wg, color='red', linewidth=2, label='West Germany')
plt.axvline(treatment_year, color='black', linestyle='--', label='Treatment Year (1990)')
plt.xlabel('Year')
plt.ylabel('Gap (GDP)')
plt.title('In-Space Placebo Test: Gaps for All Units')
plt.legend()
plt.grid()
plt.show()

print(f"RMSPE West germany: {rmspe_wg:.2f}")

```


    
    


    RMSPE West germany: 60.84


# Part f


```python
id_column = 'entity_id'
treated_unit = 'West Germany'
donor_pool = [entity for entity in df[id_column].unique() if entity != treated_unit]
treatment_year = 1990

def rmspe_calc(gap, years, split_year):
    pre = gap[years < split_year]
    post = gap[years >= split_year]
    rmspe_pre = np.sqrt(np.mean(pre ** 2))
    rmspe_post = np.sqrt(np.mean(post ** 2))
    return rmspe_pre, rmspe_post

def scm_rmspe_for_country(treated, donor_pool):
    pre_treatment = df[df['year_recorded'] < treatment_year]
    covariates = ['metric_inflation', 'trade', 'metric_education', 'metric_industry_share']

    Y1 = pre_treatment[pre_treatment[id_column] == treated]['indicator_score'].values
    Y0 = (pre_treatment[pre_treatment[id_column].isin(donor_pool)]
        [['year_recorded', id_column, 'indicator_score']]
        .pivot(index='year_recorded', columns=id_column, values='indicator_score').values)

    def objective_function(weights):
        return np.sqrt(np.mean((Y1 - Y0 @ weights) ** 2))

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1)] * len(donor_pool)
    initial_weights = np.ones(len(donor_pool)) / len(donor_pool)
    result = minimize(objective_function, initial_weights, bounds=bounds, constraints=constraints)
    weights = result.x

    pivot_table = df[df[id_column].isin(donor_pool)][['year_recorded', id_column, 'indicator_score']]
    synthetic_gdp = (pivot_table.pivot(index='year_recorded', columns=id_column, values='indicator_score').values @ weights)

    actual_df = df[df[id_column] == treated][['year_recorded', 'indicator_score']].rename(
        columns={'indicator_score': 'actual_gdp'})
    synthetic_df = pd.DataFrame({'year_recorded': df['year_recorded'].unique(), 'synthetic_gdp': synthetic_gdp})
    result_df = actual_df.merge(synthetic_df, on='year_recorded')
    result_df['gap'] = result_df['actual_gdp'] - result_df['synthetic_gdp']

    years = result_df['year_recorded'].values
    gap = result_df['gap'].values
    rmspe_pre, rmspe_post = rmspe_calc(gap, years, treatment_year)
    ratio = rmspe_post / rmspe_pre if rmspe_pre > 0 else np.nan
    return ratio

ratios = []
names = []
for country in donor_pool + [treated_unit]:
    pool = [c for c in donor_pool + [treated_unit] if c != country]
    ratio = scm_rmspe_for_country(country, pool)
    ratios.append(ratio)
    names.append(country)

idx_wg = names.index(treated_unit)
colors = ['gray'] * len(names)
colors[idx_wg] = 'red'

plt.figure(figsize=(12, 6))
plt.bar(names, ratios, color=colors)
plt.xticks(rotation=90)
plt.ylabel('Post/Pre RMSPE Ratio')
plt.title('Post/Pre RMSPE Ratios for All Units')
plt.tight_layout()
plt.show()

wg_ratio = ratios[idx_wg]
count_higher = sum([r > wg_ratio for i, r in enumerate(ratios) if i != idx_wg])
p_value = count_higher / (len(ratios) - 1)
print(f"Empirical p-value: {p_value:.3f}")
print(f"Post/Pre RMSPE Ratio for West Germany: {wg_ratio:.2f}")

```


    
    


    Empirical p-value: 0.000
    Post/Pre RMSPE Ratio for West Germany: 30.37

