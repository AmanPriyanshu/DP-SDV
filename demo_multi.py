from sdv import load_demo
from sdv.relational import HMA1

metadata, tables = load_demo(metadata=True)
model = HMA1(metadata)
model.fit(tables)
new_data = model.sample(num_rows=100)
print(new_data)
for key, item in new_data.items():
	item.to_csv('./multi_demo_results/'+key+'_normal.csv', index=False)

print("DP-TIME")

metadata, tables = load_demo(metadata=True)
model = HMA1(metadata)
model.fit(tables, eps=1e-5)
new_data = model.sample(num_rows=100)
print(new_data)
for key, item in new_data.items():
	item.to_csv('./multi_demo_results/'+key+'_dp.csv', index=False)
