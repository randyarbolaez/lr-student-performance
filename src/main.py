import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from matplotlib.ticker import MaxNLocator

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
# plt.show()


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
c = '#0055ff'

print(df.groupby(["parental_education"]).size())

for i in range(len(cat_features)):
    print(cat_features[i].count)
    sns.stripplot(ax=axes[i], x=cat_features[i], y=target, data=df, size=6, color=c)
    axes[i].set_title(f"{target} vs. {cat_features[i]}", size=12)
    axes[i].set_xlabel(f"{cat_features[i]}({df.groupby([cat_features[i]]).size()})", size=12)
    axes[i].set_ylabel(target, size=12)
    axes[i].grid()

plt.suptitle("Final Exam Score vs Categorical Features", size=20)
plt.tight_layout()
# plt.show()

df = pd.get_dummies(df, columns=cat_features, drop_first=True)
# print(df)

# correlation analysis
cmap = sns.diverging_palette(125, 28, s=100, l=65, sep=50, as_cmap=True)
fig,ax = plt.subplots(figsize=(9,8), dpi=80)
ax = sns.heatmap(pd.concat([df.drop(target,axis=1), df[target]],axis=1).corr(), annot=True, cmap=cmap)
# plt.show()

X = df.drop('final_exam_score', axis=1)
y = df['final_exam_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

y_test_actual = y_test

#normalize dataset
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

linear_reg = LinearRegression()
linear_reg.fit(X_train_scaled, y_train)
intercept_and_coefficients = pd.DataFrame(data = np.append(linear_reg.intercept_, linear_reg.coef_), index = ['Intercept'] +[col+' Coef. ' for col in X.columns], columns=[''
'Value']).sort_values("Value", ascending=False)

def model_eval(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)

    MAE = metrics.mean_absolute_error(y_test, y_pred)
    MSE = metrics.mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(MSE)
    R2_Score = metrics.r2_score(y_test, y_pred)

    return pd.DataFrame([MAE, MSE, RMSE, R2_Score], index=['MAE', 'MSE', 'RMSE', 'R2-Score'], columns=[model_name])


print(model_eval(linear_reg, X_test_scaled, y_test, 'Linear Regression'))

#K-fold cross-validation
linear_req_cv = LinearRegression()
scaler = StandardScaler()
pipeline = make_pipeline(StandardScaler(), LinearRegression())

kf = KFold(n_splits=6, shuffle=True, random_state=0)
scoring = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error', 'r2']
result = cross_validate(pipeline, X, y, cv=kf, return_train_score=True, scoring=scoring)

MAE_mean = (-result['test_neg_mean_absolute_error']).mean()
MAE_std = (-result['test_neg_mean_absolute_error']).std()
MSE_mean = (-result['test_neg_mean_squared_error']).mean()
MSE_std = (-result['test_neg_mean_squared_error']).std()
RMSE_mean = (-result['test_neg_root_mean_squared_error']).mean()
RMSE_std = (-result['test_neg_root_mean_squared_error']).std()
R2_Score_mean = result['test_r2'].mean()
R2_Score_std = result['test_r2'].std()

blah = pd.DataFrame({'Mean': [MAE_mean,MSE_mean,RMSE_mean,R2_Score_mean], 'Std': [MAE_std,MSE_std,RMSE_std,R2_Score_std]},
             index=['MAE', 'MSE', 'RMSE' ,'R2-Score'])

print(blah)

y_test_pred = linear_reg.predict(X_test_scaled)
df_comp = pd.DataFrame({'Actual': y_test_actual, 'Predicted':y_test_pred})

def compare_plot(df_comp):
    df_subset = df_comp.head(100).reset_index(drop=True)
    df_comp.reset_index(inplace=True)
    df_subset.plot(y=['Actual','Predicted'], kind='bar', figsize=(20,8), width=0.8)
    plt.title('Predicted vs. Actual Target Values for Final Exam Score', fontsize=20)
    plt.xticks(rotation=90, fontsize=6)
    plt.ylabel("Final Exam Score")
    plt.tight_layout()
    plt.show()

compare_plot(df_comp)