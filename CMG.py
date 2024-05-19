import scipy
import itertools
from scipy import stats
import sklearn


#FINDING CORRELATIONS BETWEEN BINARY VALUES 
col = ["Ålder","Vikt","Längd","Mammografiscreening","Menopaus=2","Bilateral>1","Sida","C500","C502","C503","C504","C505","Tumörstorlek","Multipel_brösttumörer","Lymfovaskulär_invasion","Tumördiagnos=1","Tumördiagnos=2","Tumördiagnos>2","Lobulär_vs_Duktal_bröstcancer","Grad","ER","ERstatus","PR","PRstatus","HER2status","Ki67","BMI"]
bini = ["Mammografiscreening","Menopaus=2","Bilateral>1","Sida","C500", "Multipel_brösttumörer","Lymfovaskulär_invasion","Tumördiagnos=1","Tumördiagnos=2","Tumördiagnos>2","Lobulär_vs_Duktal_bröstcancer","ERstatus","PRstatus","HER2status"]

correlations = []
big = []
num_prints = 0
for i, j in itertools.combinations(range(len(binary)), 2):
    corre = sklearn.metrics.matthews_corrcoef(binary[i], binary[j], sample_weight=None)
   # print(f'Correlation constant: {corre} for column {bini[i]} and {bini[j]}')
    correlations.append(corre)
    if abs(corre) > 0.5:
        print(f'Bin constants:{corre}  for {bini[i]} and {bini[j]}')
        big.append(corre)
    num_prints += 1

#FINDING CORRELATIONS BETWEEN CONTONOUS VALUES 


col = ["Ålder","Vikt","Längd","Mammografiscreening","Menopaus=2","Bilateral>1","Sida","C500","C502","C503","C504","C505","Tumörstorlek","Multipel_brösttumörer","Lymfovaskulär_invasion","Tumördiagnos=1","Tumördiagnos=2","Tumördiagnos>2","Lobulär_vs_Duktal_bröstcancer","Grad","ER","ERstatus","PR","PRstatus","HER2status","Ki67","BMI"]
colcont= ["Ålder","Vikt","Längd","Tumörstorlek","Grad","ER","PR","Ki67","BMI"]
 
ccs = []
big_cor = []
num_prints = 0
for i, j in itertools.combinations(range(len(columnss)), 2):
    corre, _ = scipy.stats.pearsonr(columnss[i], columnss[j])
    #print(f'Correlation constant: {corre} for column {col[i]} and {col[j]}')
    ccs.append(corre)
    if abs(corre) > 0.5:
        print(f'Big corr constants:{corre}  for {colcont[i]} and {colcont[j]}')
        big_cor.append(corre)
    num_prints += 1
    
