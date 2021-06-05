# Diplomski_LinReg_SVM
Diplomski rad: Predikcija korektivnih mjera u stabilizaciji sustava proizvodnog procesa

Master thesis: Corrective measures prediction in the stabilization of the production process system

Abstract:

Complexity of industrial control systems has been increasing constantly over past decades. This is mostly the case due to implementation of new technologies with the ultimate goal of achieving higher productivity results by improving process control, system availability, reliability and flexibility. Following this goal, in the era of Industry 4.0 concept, boundaries have been pushed toward Big data, Industrial internet of things, machine learning and heterogeneous networks. As a consequence, supervisory control and data acquisition (SCADA) systems manage more data, have more functions and  become more complex to use. This might not be the issue for some industries where level of automation allows unmanned process control. But in the case of industries with continuous stream processes like the steel industry with chained processes of continuous casting, rolling mills, galvanizing lines, and wire rods mills where a certain stochastic component is expected, operators are still irreplaceable. For those plants, an operator's competence and experience is crucial in maintaining process stability and continuity over time. This work focuses on operator decision support by implementing supervised machine learning models of multi-output linear regression and support vector machine used for predicting the most efficient operator’s response for a given process state in real time, achieved by analysing dataset with all operator activities to the controlled production process, and allocating the most efficient operator to be used as a reference.

Keywords: scada, hmi, classification, regression, svm, multi-output, stream, process, control, production, industry, prediction, operator

This repository contains:
1. HMI application (Hydraulic_Unit_HMI_WinCC_2021-06-05.7z).
  - developed as simulator for controlled process and operator control actions (corrective measures to the system).
2. dbHMI_tables.sql
  - tables created in dbHMI (pg database) containing tag(signal/variable) values gathered from the HMI.
3. FullDataSet.csv
  - file containig full dataframe used in thesis.
  - created by python code after connectiong to dbHMI and forming dataframe.   
4. Python code (SQL_and_Regression_and_SVM.PY).
  - forming dataframe from values stored in dbHMI
  - preparing/transforming dataframe 
    (Filtering df of single most efficient operator, deleting extreame minority class, Feature importance, oversampling...)
  - trainig multi-output linear regression (GradientBoostingRegressor)
  - training multi-output support vector machine
  - performing predictions with various df sizes
  - plotting data 
    (Variance Inflation Factor-VIF, Heat Maps, Feature importance, df boxplot, class ratio barplot...)
  - this code can be used without connection to db, by reading provided FullDataSet.csv directly.
 5. Master_thesis.pdf
    
Software used:
Windows	10 - 64bit
WinCC	v.7.5 SP1 Upd2
PostgreSQL	v.13.1
pgAdmin v.4.28
DbSchema v.8.4.2
VMware Workstation Player	v.16.0
Visual studio code v.1.55.2
Python v.3.8.8 (64bit)
Anaconda v.4.10.0
pandas v.1.1.3
sqlalchemy v.1.3.20
numpy v.1.19.2
matplotlib v.3.3.2
scikit-learn v.0.32.2


