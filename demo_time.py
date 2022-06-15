from DPSDV.demo import load_timeseries_demo
from DPSDV.timeseries import PAR

data = load_timeseries_demo()
entity_columns = ['Symbol']
context_columns = ['MarketCap', 'Sector', 'Industry']
sequence_index = 'Date'

model = PAR(entity_columns=entity_columns,context_columns=context_columns,sequence_index=sequence_index)
model.fit(data)
new_data = model.sample(1)
new_data.to_csv('./time_demo_results/normal.csv', index=False)

model = PAR(entity_columns=entity_columns,context_columns=context_columns,sequence_index=sequence_index)
model.fit(data, noise_multiplier=1e-3, max_grad_norm=1.0)
new_data = model.sample(1)
new_data.to_csv('./time_demo_results/dp.csv', index=False)