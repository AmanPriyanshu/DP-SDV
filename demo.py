from sdv.demo import load_tabular_demo
from sdv.lite import TabularPreset
from sdv.tabular import GaussianCopula, CopulaGAN, CTGAN, TVAE
import warnings
warnings.filterwarnings("ignore")

metadata, data = load_tabular_demo('student_placements', metadata=True)
print(data.head())

# print('-'*40,'\n')
# model = TabularPreset(name='FAST_ML', metadata=metadata, eps=1e5)
# model.fit(data)
# synthetic_data = model.sample(num_rows=100)
# print('FAST_ML')
# print(synthetic_data.head())

# print('-'*40, '\n')
# model = GaussianCopula()
# model.fit_dp(data, eps=1e5)
# synthetic_data = model.sample(num_rows=100)
# print('Gaussian Copula')
# print(synthetic_data.head())

# print('-'*40, '\n')
# model = CTGAN()
# model.fit(data, noise_multiplier=1.4, max_grad_norm=1.0)
# synthetic_data = model.sample(num_rows=100)
# print('CT-GAN')
# print(synthetic_data.head())

# print('-'*40, '\n')
# model = CopulaGAN()
# model.fit(data, noise_multiplier=1.4, max_grad_norm=1.0)
# synthetic_data = model.sample(num_rows=100)
# print('Copula-GAN')
# print(synthetic_data.head())

print('-'*40, '\n')
model = TVAE()
model.fit(data, noise_multiplier=1e-3, max_grad_norm=1.0)
synthetic_data = model.sample(num_rows=100)
print('TVAE')
print(synthetic_data.head())

