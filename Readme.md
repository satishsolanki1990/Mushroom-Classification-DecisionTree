# Mushroom Classification

---

## Content

- [Objective](#obj)
- [Data Summary and Preprocessing](#data)
- [1. Decision Tree](#dtree)
- [2. Random Forest](#randforest)
- [3. Adaboost Algorithm](#adaboost)
- [Results and Discussion](#results)
- [Conclusion](#conclusion)

<details open>
<summary>
<a name="obj"><b style="font-size:20px">
Objective
</b></a>

The objective of the project is to classify mushrooms into edible and poisonous categories
using:

1. Decision Tree 
2. Random Forest 
3. Adaboost
</summary>
</details>

<details>
<summary> <a name="data"><b style="font-size:20px">
Data Summary and Preprocessing
</b></a>
</summary>

The data is taken directly from the commonly used UCI Machine Learning
Repository. [link](https://archive.ics.uci.edu/ml/datasets/mushroom)

Here is a short description of each train and validation split:

1. Train Set (pa3 train.csv): Includes 4874 rows (samples). Each sample contains the class (poisonous
or edible) with 22 categorical features (split into one-hot vectors for total of 117 features) related to
the mushroom’s properties.

2. Validation Set (pa3 val.csv): Includes 1625 rows. Each row obeys the same format given for the
train set. This set will be used to see the performance of the models.

</details>


<details>
<summary><a name="dtree"><b style="font-size:20px">
1: Decision Tree 
</b></a>
</summary>

</details>

<details>
<summary><a name="randforest"><b style="font-size:20px">
2: Random Forest 
</b></a>
</summary>
First, the tree was made with a depth of 2 which was used as a baseline to determine accuracy. 
The depth 2 tree had a training accuracy of 0.953 and a validation accuracy of 0.964. 
Interestingly, the validation accuracy was higher than the training accuracy, 
and this was observed as depths increased as well. One explanation for this is that the 
validation data is cleaner, and may be easier to classify whereas the training data by 
comparison may have been more difficult to create a model with.  
Unsurprisingly, the accuracy for both train and validation improved as the depth of the 
tree increased. This is because a decision tree with more splits is able to classify more 
accurately based on the principle of measuring similarities with more features. 
After depth 5, 100% accuracy was obtained on training and validation data. Therefore,
in the interest of keeping the trees as small as possible, 
depth 6 was determined to be the best for accuracy.

</details>

<details>
<summary><a name="adaboost"><b style="font-size:20px">
3: Adaboost Algorithm
</b></a>
</summary>

...

</details>


<details>
<summary><a name="results"><b style="font-size:20px">
Results and Discussion
</b></a>
</summary>

...

</details>

<details>
<summary><a name="conclusion"><b style="font-size:20px">
Conclusion
</b></a>
</summary>

...

</details>