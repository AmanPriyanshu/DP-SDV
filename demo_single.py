from DPSDV.demo import load_tabular_demo
from DPSDV.lite import TabularPreset
from DPSDV.tabular import GaussianCopula, CopulaGAN, CTGAN, TVAE, MWEMSynthesizer
from DPSDV.evaluation import evaluate
from DPSDV.metrics.tabular import BNLikelihood, BNLogLikelihood, GMLogLikelihood, LogisticDetection, SVCDetection, NumericalLR, NumericalMLP, NumericalSVR
import warnings
import pandas as pd
warnings.filterwarnings("ignore")

metadata, data = load_tabular_demo('student_placements', metadata=True)
print(data.head())

def compute_results(synthetic_data, data):
	r = evaluate(synthetic_data, data, metrics=['CSTest', 'KSTest'], aggregate=False)
	p = {'CSTest': r.raw_score[0], 'KSTest': r.raw_score[1]}
	metrics_arr = [BNLikelihood, BNLogLikelihood, GMLogLikelihood]
	metrics_arr_ = [LogisticDetection, SVCDetection]
	for m in metrics_arr:
		try:
			p.update({m.__name__: m.compute(data.fillna(0), synthetic_data.fillna(0))})
		except:
			p.update({m.__name__: None})
	for m in metrics_arr_:
		try:
			p.update({m.__name__: m.compute(data, synthetic_data)})
		except:
			p.update({m.__name__: None})
	metrics_arr__ = [NumericalLR, NumericalMLP, NumericalSVR]
	for m in metrics_arr__:
		try:
			p.update({'PRIV_METRIC_'+m.__name__: m.compute(data.fillna(0),synthetic_data.fillna(0),key_fields=['second_perc', 'mba_perc', 'degree_perc'],sensitive_fields=['salary'])})
		except:
			try:
				p.update({'PRIV_METRIC_'+m.__name__: m.compute(data.fillna(0),synthetic_data.fillna(0),key_fields=['age', 'sex', 'educ', 'race'],sensitive_fields=['married'])})
			except:
				p.update({'PRIV_METRIC_'+m.__name__: None})
	return p

performance = []
print('-'*40,'\n')
model = TabularPreset(name='FAST_ML', metadata=metadata, eps=1)
model.fit(data)
synthetic_data = model.sample(num_rows=100)
print('FAST_ML-DP')
r = {'name':'FAST_ML-DP'}
r.update(compute_results(synthetic_data, data))
performance.append(r)
synthetic_data.to_csv("./single_demo_results/FAST_ML_dp.csv", index=False)

print('-'*40,'\n')
model = TabularPreset(name='FAST_ML', metadata=metadata)
model.fit(data)
synthetic_data = model.sample(num_rows=100)
print('FAST_ML')
r = {'name':'FAST_ML'}
r.update(compute_results(synthetic_data, data))
performance.append(r)
synthetic_data.to_csv("./single_demo_results/FAST_ML_normal.csv", index=False)


df = pd.read_csv("./DPSDV/data/PUMS_california_demographics/data.csv")
df = df.drop(["income"], axis=1)
print('-'*40,'\n')
model = MWEMSynthesizer(epsilon=1.0)
model.fit(df)
synthetic_data = model.sample(num_rows=100)
print('MWEM-DP')
r = {'name':'MWEM-DP'}
r.update(compute_results(synthetic_data, df))
performance.append(r)
synthetic_data.to_csv("./single_demo_results/MWEM_dp.csv", index=False)

print('-'*40, '\n')
model = GaussianCopula()
model.fit_dp(data, eps=1)
synthetic_data = model.sample(num_rows=100)
print('Gaussian Copula-DP')
r = {'name':'Gaussian Copula-DP'}
r.update(compute_results(synthetic_data, data))
performance.append(r)
synthetic_data.to_csv("./single_demo_results/GaussianCopula_dp.csv", index=False)

print('-'*40, '\n')
model = GaussianCopula()
model.fit_dp(data)
synthetic_data = model.sample(num_rows=100)
print('Gaussian Copula')
r = {'name':'Gaussian Copula'}
r.update(compute_results(synthetic_data, data))
performance.append(r)
synthetic_data.to_csv("./single_demo_results/GaussianCopula_normal.csv", index=False)

print('-'*40, '\n')
model = CTGAN()
model.fit(data, noise_multiplier=1.4, max_grad_norm=1.0)
synthetic_data = model.sample(num_rows=100)
print('CT-GAN-DP')
r = {'name':'CT-GAN-DP'}
r.update(compute_results(synthetic_data, data))
performance.append(r)
print(synthetic_data.head())
synthetic_data.to_csv("./single_demo_results/CTGAN_dp.csv", index=False)

print('-'*40, '\n')
model = CTGAN()
model.fit(data)
synthetic_data = model.sample(num_rows=100)
print('CT-GAN')
r = {'name':'CT-GAN'}
r.update(compute_results(synthetic_data, data))
performance.append(r)
synthetic_data.to_csv("./single_demo_results/CTGAN_normal.csv", index=False)

print('-'*40, '\n')
model = CopulaGAN()
model.fit(data, noise_multiplier=1.4, max_grad_norm=1.0)
synthetic_data = model.sample(num_rows=100)
print('Copula-GAN-DP')
r = {'name':'Copula-GAN-DP'}
r.update(compute_results(synthetic_data, data))
performance.append(r)
synthetic_data.to_csv("./single_demo_results/CopulaGAN_dp.csv", index=False)

print('-'*40, '\n')
model = CopulaGAN()
model.fit(data)
synthetic_data = model.sample(num_rows=100)
print('Copula-GAN')
r = {'name':'Copula-GAN'}
r.update(compute_results(synthetic_data, data))
performance.append(r)
synthetic_data.to_csv("./single_demo_results/CopulaGAN_normal.csv", index=False)

print('-'*40, '\n')
model = TVAE()
model.fit(data, noise_multiplier=1e-3, max_grad_norm=1.0)
synthetic_data = model.sample(num_rows=100)
print('TVAE-DP')
r = {'name':'TVAE-DP'}
r.update(compute_results(synthetic_data, data))
performance.append(r)
synthetic_data.to_csv("./single_demo_results/TVAE_dp.csv", index=False)

print('-'*40, '\n')
model = TVAE()
model.fit(data)
synthetic_data = model.sample(num_rows=100)
print('TVAE')
r = {'name':'TVAE'}
r.update(compute_results(synthetic_data, data))
performance.append(r)
synthetic_data.to_csv("./single_demo_results/TVAE_normal.csv", index=False)

df = {}
for key in performance[0].keys():
	df.update({key: []})
for key in performance[0].keys():
	for row in performance:
		df[key].append(row[key])

df = pd.DataFrame(df)
df.to_csv("performance_demo_single", index=False)