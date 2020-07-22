# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 14:43:03 2020

@author: gy605
"""

import pandas as pd 

from scipy.stats import pearsonr

dataset = pd.read_csv("general_data.csv")

dataset1 = pd.read_csv("general_data.csv")

dataset["Attrition"] = dataset["Attrition"].astype('category')
dataset["Attrition"] = dataset["Attrition"].cat.codes

cor_mat = dataset.corr()

stats1,p1 = pearsonr(dataset.Attrition,dataset.Age)
print("stats1 , p1 for Attrition and Age : " ,stats1,p1)
stats2,p2 = pearsonr(dataset.Attrition,dataset.DistanceFromHome)
print("stats2 , p2 for Attrition and DistanceFromHome : " ,stats2,p2)
stats3,p3 = pearsonr(dataset.Attrition,dataset.Education)
print("stats3 , p3 for Attrition and Education : " ,stats3,p3)
stats4,p4 = pearsonr(dataset.Attrition,dataset.JobLevel)
print("stats4 , p4 for Attrition and JobLevel : " ,stats4,p4)
stats5,p5 = pearsonr(dataset.Attrition,dataset.MonthlyIncome)
print("stats5 , p5 for Attrition and MonthlyIncome : " ,stats5,p5)
stats6,p6 = pearsonr(dataset.Attrition,dataset.PercentSalaryHike)
print("stats6 , p6 for Attrition and PercentSalaryHike : " ,stats6,p6)
stats7,p7 = pearsonr(dataset.Attrition,dataset.YearsAtCompany)
print("stats7 , p7 for Attrition and YearsAtCompany : " ,stats7,p7)
stats8,p8 = pearsonr(dataset.Attrition,dataset.YearsSinceLastPromotion)
print("stats8 , p8 for Attrition and YearsSinceLastPromotion : " ,stats8,p8)

from scipy.stats import chi2_contingency
chitable1 = pd.crosstab(dataset1.Attrition, dataset1.BusinessTravel)
print(chitable1)
stats9,p9,dof,expected = chi2_contingency(chitable1)
print("stats9,p9 for Attrition and BusinessTravel is :", stats9,p9) 

chitable2 = pd.crosstab( dataset1.Attrition, dataset1.Department)
print(chitable2)
stats10,p10,dof,expected = chi2_contingency(chitable2)
print("stats10,p10 for Attrition and Department is :", stats10,p10) 

chitable3 = pd.crosstab( dataset1.Attrition, dataset1.EducationField)
print(chitable3)
stats11,p11,dof,expected = chi2_contingency(chitable3)
print("stats11,p11 for Attrition and EducationField is :", stats11,p11) 