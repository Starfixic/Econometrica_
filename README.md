# Econometrica
Конспект эконометрика.
Условные обозначения:
То, что нужно скопировать, не выделяю. То, что нужно заменить на другие данные выделяю желтым.
Серым выделяю редко нужную фигню.
Питон для всех моделей:
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.tools import add_constant
from statsmodels.iolib.summary2 import summary_col, summary_params
from scipy.stats import t # t-распределение
from scipy.stats import norm # нормальное распределение
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_params
from scipy.stats import t # t-распределение
from scipy.stats import f # F-распределение
from scipy.stats import chi2 # критические значения chi2
# подключение датасета.
df = pd.read_csv('НАЗВАНИЕ ФАЙЛА.csv', na_values=(' ', '', '  '))
# посмотрим на первые пять строк датасета
df.head(n=5)
# размеры датасета
df.shape
# информация о датасете
df.info()
# Если нужно записать квадрат параметра (потом в модели вписываем I)
I=df.ПАРАМЕТР**2

LPM модель (линейная регрессия)
Запись модели: 
approve=β0+β1mortno+β2unem+β3dep+β4male+β5married+β6yjob+β7self+u
ЛИБО: P(approve=1)=β0+β1mortno+β2unem+β3dep+β4male+β5married+β6yjob+β7self
Питон:
Включаем LPM модель
mod_lpm = smf.ols(formula='approve~параметр+…+параметр', data=df)
# подгонка модели
res_lpm_hc = mod_lpm.fit(cov_type='HC3')
# результаты регрессии
print(res_lpm_hc.summary(slim=True))
# Сравнение моделей (Опционально)
res_lpm_ols = mod_lpm.fit(cov_type='nonrobust')
print(summary_col([res_lpm_hc, res_lpm_ols], model_names=['Robust', 'Non-robust'], stars=True))
# Коэфициенты модели с округлением до 3-х десятичных знаков
res_lpm_hc.params.round(3)
 
ИНТЕРПРЕТАЦИЯ
- Если бинарная пишем так: для регрессора _________ разница вероятностей между 0 и 1 равна __
- Если количественная пишем так: при увеличении регрессора _________ на единицу, вероятность успеха уменьшится или увеличится на _________
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
T-test (тест значимости коэф. LPM модели)
Запись теста:

Вариант 1: – Т-статистика - делим бету (Coef.) на стандартную ошибку (Std.Err). 

     - Т-критическая – находится по таблице распределения Стьюдента, по пересечению уровня значимости α (обычно равно 1%,5% или 10%) и степеней несвободы df (df=n (number of observations) – k (количество коэфициентов без constant или intersept) - 1)
Сравниваем. Если |tst| > tcr, то Х при этом регресоре значим. (tst берём по МОДУЛЮ!!!)

Вариант 2. Через P – значения. Если P< α, то коэфициент значим.
 
Питон:
Считаем отдельно каждый параметр для варианта 1:
#  t-статистика для каждого коэффициента с округлением до 3-х десятичных знаков
np.round(res_lpm_hc.tvalues, 3)
# Число наблюдений в модели (number of observations)
res_lpm_hc.nobs
# число регрессоров (количество коэфициентов без constant или intersept)
res_lpm_hc.df_model
# степени свободы (df)
res_lpm_hc.df_resid
# Вывести tcr с округлением до 3-х десятичных знаков при α = 1%
np.round(t.ppf(q=1-0.01/2, df=res_lpm_hc.df_resid), 3)
# P-значения для t-статистик с округленим до 3-х десятичных знаков
res.pvalues.round(3)
# Сначала обязательно должна быть прогнана LPM модель
# Показываем результаты t-теста для коэффициентов (неробастные s.e. ( _ols))
res_lpm_ols = mod_lpm.fit(cov_type='nonrobust')
summary_params(res_lpm_ols, alpha=0.01)
# Выводим значимость коэффициентов
df_ols = np.round(summary_params(res_lpm_ols, alpha=0.01), 3)
df_ols['significance'] = df_ols.apply(lambda x: 'Значим' if x['P>|t|']<0.01 else 'Незначим', axis=1)
df_ols
 
# Показываем результаты t-теста для коэффициентов (робастные s.e. ( _hc))
summary_params(res_lpm_hc, alpha=0.01)
# Выводим значимость коэффициентов
df_hc = np.round(summary_params(res_lpm_hc, alpha=0.01), 3)
df_hc['significance'] = df_hc.apply(lambda x: 'Значим' if x['P>|t|']<0.01 else 'Незначим', axis=1)
df_hc
 
# Можем вывести значимость отдельных коэффициентов (робастные s.e.)
res_lpm_hc.t_test('Коэфициент=0, Коэфициент2=0')
 
ИНТЕРПРЕТАЦИЯ (Значимость)
 
Если |tst| > tcr, то Х при этом регресоре значим. (tst берём по МОДУЛЮ!!!)

Через P – значения. Если P< α, то коэфициент значим.
 
 
 
 
 
F-test (тест значимости всей LPM модели)
Запись модели:
 
Два варианта, как и в t-test. 
Вариант 1: Через сравнение F-статистики и F-критического. Прогоняем LPM модель, там есть F-statistic. Затем считаем F-критическое одной строчкой и сравнивааем. Если Fst>Fcr, регрессия значима.
Вариант 2: Через P, как и в t-тесте. Когда прогоняем LPM модель, мы получаем Prob (по F-statistic), затем сравниваем с уровнем значимости α (обычно равно 1%,5% или 10%). Если P< α, регрессия значима.
Питон:
Сначала нужно прогнать LPM модель:
mod_lpm = smf.ols(formula='approve~параметр+…+параметр', data=df)
# подгонка модели
res_lpm_hc = mod_lpm.fit(cov_type='HC3')
# результаты регрессии (тут смотрим F-statistic и Prob (по F-statistic))
print(res_lpm_hc.summary(slim=True))
# F - статистика и P-значение (РОБАСТНЫЕ)
np.round(res_lpm_hc.fvalue, 3), np.round(res_lpm_hc.f_pvalue, 3)

# тестовая статистика и P-значение (НЕ РОБАСТНЫЕ)
res_lpm_ols = mod_lpm.fit(cov_type='nonrobust')
res_lpm_ols.fvalue.round(3), res_lpm_ols.f_pvalue.round(3)
 
# Выводи F-критическое, при α = 10%
f.ppf(q=1-0.1, dfn=res_lpm_hc.df_model, dfd=res_lpm_hc.df_resid).round(3)
 
 
ИНТЕРПРЕТАЦИЯ
Если Fst> Fcr, то регрессия значима.

Через P – значения. Если P(по F-statistic)< α, то регрессия значима.
 
 
 
 
 
 
 
 
 
 
 
 
 
 
F-test (тест значимости отдельных коэф. LPM модели)
Запись модели:
 
Разница только в df1=J (количество выбранных коэфициентов)
Два варианта, как и в t-test. 
Вариант 1: Через сравнение F-статистики и F-критического. Прогоняем LPM модель, там есть F-statistic. Затем считаем F-критическое одной строчкой и сравнивааем. Если Fst>Fcr, регрессия значима. 
Вариант 2: Через P, как и в t-тесте. Когда прогоняем LPM модель, мы получаем Prob (по F-statistic), затем сравниваем с уровнем значимости α (обычно равно 1%,5% или 10%). Если P< α, регрессия значима.
Питон:
Сначала нужно прогнать LPM модель:
mod_lpm = smf.ols(formula='approve~параметр+…+параметр', data=df)
# подгонка модели
res_lpm_hc = mod_lpm.fit(cov_type='HC3')
# результаты регрессии (тут смотрим F-statistic и Prob (по F-statistic))
print(res_lpm_hc.summary(slim=True))
# F-статистика и P-значение
f.ppf(q=1-0.05, dfn=2, dfd=res_lpm_hc.df_resid).round(3)
 
# F-критическое и P-значение (РОБАСТНЫЕ)
I=df.appinc**2
# тестовая статистика, P-значение и степени свободы
print(res_lpm_hc.f_test('appinc=I=0'))
# тестовая статистика, P-значение и степени свободы
print(res_lpm_hc.wald_test('appinc=I=0', use_f=True))

# F-критическое и P-значение (НЕ РОБАСТНЫЕ)
# подгонка модели
res_lpm_ols = mod_lpm.fit(cov_type='nonrobust')
I=df.appinc**2
# тестовая статистика, P-значение и степени свободы
print(res_lpm_ols.f_test('appinc=I=0'))
 
 
ИНТЕРПРЕТАЦИЯ
Если Fst> Fcr, то регрессия значима.

Через P – значения. Если P< α, то регрессия значима.
 
 
 
 
 
 
 
 
Probit модель (Φ)
Запись модели:
P(approve=1)=Φ(β0+β1appinc+β2mortno+β3unem+β4dep+β5male+β6married+β7yjob+β8self)
ЛИБО:
probit(P(approve=1))=β0+β1appinc+β2mortno+β3unem+β4dep+β5male+β6married+β7yjob+β8self
 
Питон:
Включаем Probit модель
mod = smf.probit(formula='approve~ параметр+…+параметр', data=df)
# подгонка модели
res = mod.fit()
# результаты регрессии
res.summary()
# коэффициенты подогнанной модели с округлением
res.params.round(3)
 
ИНТЕРПРЕТАЦИЯ (Только знак при регрессоре)
- Если бинарная пишем так: для людей с регрессором (женатых, трудоустроенных) _________ вероятность успеха увеличивается (уменьшается), чем для людей с обратным значением (неженатых, нетрудоустроенных).
- Если количественная пишем так: при увеличении регрессора _________ на единицу, вероятность успеха умешается или увеличивается (смотрим на знак).
 
Logit модель (Λ)
Запись модели:
P(approve=1)=Λ(β0+β1appinc+β2mortno+β3unem+β4dep+β5male+β6married+β7yjob+β8self)
ЛИБО
logit(P(approve=1))=β0+β1appinc+β2mortno+β3unem+β4dep+β5male+β6married+β7yjob+β8self
ЛИБО 
logit(P(approve=1))=logP(approve=1)/(1−P(approve=1))=logP(approve=1)/(P(approve=0))
Питон:
Включаем Probit модель
mod = smf.logit(formula='approve~параметр+…+параметр', data=df)
# подгонка модели
res = mod.fit()
# результаты регрессии
res.summary()
# коэффициенты подогнанной модели с округлением
res.params.round(3)
 
ИНТЕРПРЕТАЦИЯ (Отношение шансов)
- Если бинарная пишем так: для людей с регрессором (женатых, трудоустроенных) _________ отношение шансов больше (меньше) на __%, чем для людей с обратным значением (неженатых, нетрудоустроенных).
- Если количественная пишем так: при увеличении регрессора _________ на единицу, отношение шансов уменьшается (увеличивается) на __%.
 
 
 
Z-test (тест значимости коэф. Probit и Logit моделей)
Полностью идентичен T-тесту, только Z-критическая – находится по таблице нормального распределения.
 
Питон:
Считаем отдельно каждый параметр для варианта 1:
#  t-статистика для каждого коэффициента с округлением до 3-х десятичных знаков
np.round(res_lpm_hc.tvalues, 3)
# Число наблюдений в модели (number of observations)
res_lpm_hc.nobs
# число регрессоров (количество коэфициентов без constant или intersept)
res_lpm_hc.df_model
# степени свободы (df)
res_lpm_hc.df_resid
# Вывести zcr с округлением до 3-х десятичных знаков при α = 1%
sign_level = 0.01
norm.ppf(q=1-sign_level/2)
# P-значения для t-статистик с округленим до 3-х десятичных знаков
res.pvalues.round(3)
 
# Показываем результаты z-теста для коэффициентов (неробастные s.e. ( _ols))
res_lpm_ols = mod_lpm.fit(cov_type='nonrobust')
summary_params(res_lpm_ols, alpha=0.01)
# Выводим значимость коэффициентов
df_ols = np.round(summary_params(res_lpm_ols, alpha=0.01), 3)
df_ols['significance'] = df_ols.apply(lambda x: 'Значим' if x['P>|t|']<0.01 else 'Незначим', axis=1)
df_ols
 
# Показываем результаты t-теста для коэффициентов (робастные s.e. ( _hc))
summary_params(res_lpm_hc, alpha=0.01)
# Выводим значимость коэффициентов
df_hc = np.round(summary_params(res_lpm_hc, alpha=0.01), 3)
df_hc['significance'] = df_hc.apply(lambda x: 'Значим' if x['P>|t|']<0.01 else 'Незначим', axis=1)
df_hc
 
# Можем вывести значимость отдельных коэффициентов (робастные s.e.)
res_lpm_hc.t_test('Коэфициент=0, Коэфициент2=0')
 
ИНТЕРПРЕТАЦИЯ (Значимость)
 
Если |zst| > zcr, то Х при этом регресоре значим. (tst берём по МОДУЛЮ!!!)

Через P – значения. Если P< α, то коэфициент значим.
 
 
 
 
 
 
 
 
 
 
 
 
LR-test (тест значимости всей Probit или Logit модели)
LR = 2(logLlikelihood – logLnull)

Если LR>x2df(α), то регрессия значима
Если P< α, регрессия значима.
 
Питон:
Считаем отдельно каждый параметр:
# Число наблюдений, по которым была оценена модель
res.nobs
# Loglikelihood
res.llf.round(3)
# Lognull
res.llnull.round(3)
 
# Тестовая статистика LR-теста и её P-значение с округленим
res.llr.round(3), res.llr_pvalue.round(3)
# Степени свободы для х2 распределения
res.df_model
# Критические значения x2df(α)
chi2.ppf(q=1-0.05, df=res.df_model).round(3)
 
ИНТЕРПРЕТАЦИЯ (Значимость)
 
Если LR>x2df(α), то регрессия значима
Если P< α, регрессия значима.
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
LR-test (тест значимости отдельных коэф. Probit или Logit модели)
LR = 2(logLlikelihood – logLr)

Если LR>x2df(α), то регрессия значима
Если P< α, регрессия значима.
 
Питон:
Считаем отдельно каждый параметр:
# Число наблюдений, по которым была оценена модель
res.nobs
# Loglikelihood
res.llf.round(3)
# Lognull
res.llnull.round(3)
 
# подгонка модели с ограничениями
df_mod = df[['approve','appinc','mortno','unem','dep', 'male', 'married', 'yjob', 'self']].dropna()
mod_r = smf.logit(formula ='approve~mortno+unem+dep+male+married+yjob+self', data = df_mod)
res_r = mod_r.fit()
res_r.nobs # число наблюдений, на которых была подогнана модель
 
# Тестовая статистика LR-теста с округленим (Значение LR)
lr_stat=2*(res.llf-res_r.llf)
lr_stat.round(3)
# Степени свободы для х2 распределения
res.df_model-res_r.df_model
# P-значение тестовой статистики LR-теста с округленим
lr_pvalue = chi2.sf(lr_stat, df=res.df_model-res_r.df_model)
lr_pvalue.round(3)
# Критические значения x2df(α)
sign_level = 0.01 # уровень значимости
chi2.ppf(q=1-sign_level, df=res.df_model-res_r.df_model).round(3)
 
ИНТЕРПРЕТАЦИЯ (Значимость)
 
Если LR>x2df(α), то регрессия значима
Если P< α, регрессия значима.
 
 
 
