from sdv.demo import load_tabular_demo
from sdv.lite import TabularPreset

metadata, data = load_tabular_demo('student_placements', metadata=True)
print(data.head())
model = TabularPreset(name='FAST_ML', metadata=metadata, eps=1e5)
model.fit(data)
synthetic_data = model.sample(num_rows=100)
print(synthetic_data.head())