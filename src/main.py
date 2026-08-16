import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("./student_performance_dataset.csv")
df.info()

# dropping final_grade bc its the same as final_exam_score
df.drop('final_grade', axis = 1, inplace=True)
df.drop('gender', axis = 1, inplace=True)
df.drop('student_id', axis = 1, inplace=True)
df.info()

df['parental_education'] = df['parental_education'].fillna("None")
print(df['parental_education'].unique())

num_features = ['study_time_hours', 'attendance_percent', 'sleep_hours', 'previous_grade']
cat_features = ['parental_education', 'internet_access','extracurricular_activities', 'part_time_job']


#finding outliers using boxplot
sns.set_style('darkgrid')
colors = ['#0055ff', '#ff7000', '#23bf00']
CustomPalette = sns.set_palette(sns.color_palette(colors))

fig, ax = plt.subplots(2, 2, figsize=(15,7),dpi=100)

for i,col in enumerate(num_features):
    x = i//2
    y = i%2
    print(col)
    sns.boxplot(data=df, y=col, ax=ax[x,y])
    ax[x,y].yaxis.label.set_size(15)
plt.tight_layout()    

fig, ax = plt.subplots(2, 2, figsize=(15,7),dpi=100)
for i,col in enumerate(cat_features):
    x = i//2
    y = i%2
    print(col)
    sns.boxplot(data=df, x=col, y='final_exam_score', ax=ax[x,y])
    ax[x,y].yaxis.label.set_size(15)

plt.tight_layout()    
plt.show()


########### Final Exam Score vs Numerical Features Bivariate Analysis
fig, ax = plt.subplots(nrows=2 ,ncols=2, figsize=(10,10), dpi=90)
target = 'final_exam_score'
c = '#0055ff'

for i in range(len(num_features)):
    row = i // 2
    col = i % 2
    ax[row,col].scatter(df[num_features[i]], df[target], color=c, edgecolors='w', linewidth=0.25)
    ax[row,col].set_title(f"{target} vs. {num_features[i]}", size=12)
    ax[row,col].set_xlabel(num_features[i], size=12)
    ax[row,col].set_ylabel(target, size=12)
    ax[row,col].grid()

plt.suptitle("Final Exam Score vs Numerical Features", size=20)
# plt.tight_layout()

########### Final Exam Score vs Categorical Features Bivariate Analysis
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(10,10), dpi=90)
target = 'final_exam_score'
c = '#0055ff'

print(df.groupby(["parental_education"]).size())

for i in range(len(cat_features)):
    print(cat_features[i].count)
    sns.stripplot(ax=axes[i], x=cat_features[i], y=target, data=df, size=6, color=c)
    axes[i].set_title(f"{target} vs. {cat_features[i]}", size=12)
    axes[i].set_xlabel(f"{cat_features[i]}({df.groupby([cat_features[i]]).size()})", size=12)
    axes[i].set_ylabel(target, size=12)
    axes[i].grid()

# plt.suptitle("Final Exam Score vs Categorical Features", size=20)
# plt.tight_layout()
# plt.show()