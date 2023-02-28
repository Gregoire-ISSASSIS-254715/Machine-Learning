# SEND TO 183892@vutbr.cz

##############################################
#################### LAB1 ####################
##############################################

# Import libraries
'''when importing numpy, the common convention is
import numpy as np'''
import numpy
''''delete unused imports'''
import matplotlib
from matplotlib import pyplot
import pandas as pd

##############################################
# Exercise 1.4.1
print("Exercise 1.4.1 :")

# Firstly, we print "x" from 1 to 5
for y in range(6):
    print(y * "x")

# Then, we print "x" from 4 to 1
while y > 0:
    y -= 1
    print(y * "x")
##############################################


##############################################
# Exercise 1.4.2
print("\nExercise 1.4.2 :")

# String initialization
input_str = "n45as29@#8ss6"
sum = 0

# Go through all element of the string, if they are digits, add them in the sum of numbers
for x in input_str:
    if x.isdigit():
        sum += int(x)

# Finally, we display the results
print("The sum of all the numbers in this string is : ", sum)
##############################################


##############################################
# Exercise 1.4.3
print("\nExercise 1.4.3 :")

# Firstly, we let the user enter a number
n = int(input("Please, enter a number :"))
originalNumber = n

# Intermediary List and final String initialization
reversedBinaryString = []
binaryString = ""

# Then, we convert the input in binary
while n != 0:
    reversedBinaryString.append(n % 2)
    n = n // 2

# We reverse the intermediary List into the final String to get the values in the good order
for i in reversed(reversedBinaryString):
    binaryString += str(i)

# Finally, we display the results
print("The binary string of", originalNumber, " is :", binaryString)
##############################################


##############################################
# Exercise 1.5-1
print("\nExercise 1.5-1 :")


def fibonaci(upper_threshold: int) -> list:
    # Firstly, we implement a list with mandatory values
    list = [0, 1]

    # Then we compute the sum of the two elements with the highest indexes of the list
    while True:
        x = len(list) - 1
        y = x - 1
        sum = list[x] + list[y]

        # If the sum is not higher than the user input, we add it at the end of the list
        '''insted of using while True, it would be better to put this condition into while condition  '''
        if sum < upper_threshold:
            list.append(sum)

        # If the sum is higher than the user input, we stop the algorithm
        else:
            break

    # Finally, we display the values
    print("\nMax value :", upper_threshold)
    print("Fibonaci list :", list)
##############################################


##############################################
# Exercise 1.5-2
print("\nExercise 1.5-2 :")

''' your solution if fine a works perfectly, more elengant would be to use ''.join() function, so one of the possible solution would be:
 def display_as_digi(number: int):
  numbers = {1: ['  x', '  x', '  x', '  x', '  x'],
             2: ['xxx', '  x', 'xxx', 'x  ', 'xxx'],
             3: ['xxx', '  x', 'xxx', '  x', 'xxx'],
             4: ['x x', 'x x', 'xxx', '  x', '  x'],
             5: ['xxx', 'x  ', 'xxx', '  x', 'xxx'],
             6: ['xxx', 'x  ', 'xxx', 'x x', 'xxx'],
             7: ['xxx', '  x', '  x', '  x', '  x'],
             8: ['xxx', 'x x', 'xxx', 'x x', 'xxx'],
             9: ['xxx', 'x x', 'xxx', '  x', '  x']}

  full_text = ''
  number_str = str(number)

  for row in range(5):
       full_text += ' '.join([numbers[int(digit)][row] for digit in number_str])
       full_text += '\n'

  print(full_text)


display_as_digi(568)
 
 
 '''



def display_as_digi(number: int) -> None:
    # Firstly, we get the list of all the numbers in the integer, in the right order
    integersList = []

    while number > 0:
        if number < 10:
            integersList.insert(0, number)
            break

        else:
            integersList.insert(0, number % 10)
            number = number // 10

    # We print an empty line in order to display correctly the digits
    print()

    # Then, we initialize the digits dictionnary
    numbers = {0: ['xxx', 'x x', 'x x', 'x x', 'xxx'],
               1: ['  x', '  x', '  x', '  x', '  x'],
               2: ['xxx', '  x', 'xxx', 'x  ', 'xxx'],
               3: ['xxx', '  x', 'xxx', '  x', 'xxx'],
               4: ['x x', 'x x', 'xxx', '  x', '  x'],
               5: ['xxx', 'x  ', 'xxx', '  x', 'xxx'],
               6: ['xxx', 'x  ', 'xxx', 'x x', 'xxx'],
               7: ['xxx', '  x', '  x', '  x', '  x'],
               8: ['xxx', 'x x', 'xxx', 'x x', 'xxx'],
               9: ['xxx', 'x x', 'xxx', '  x', 'xxx']}

    # Finally, we run into the dictionnary
    for i in range(0, 5):
        for j in integersList:
            print(numbers[j][i], end="  ")
        print()
##############################################


##############################################
# Exercise 2.1
print("\nExercise 2.1 :")

# Declare the given array in the exercise

'''numpyArray = numpy.arrage(25,0,-1).reshape((5,5))'''
numpyArray = numpy.array([
    [25, 24, 23, 22, 21],
    [20, 19, 18, 17, 16],
    [15, 14, 13, 12, 11],
    [10, 9, 8, 7, 6],
    [5, 4, 3, 2, 1]
])
''' this does not work for me'''
# Let the user enter the threshold and array size inputs
userThreshold = int(input("Please, enter a threshold number : "))
userArray = []
arraySize = int(input("Size of array : "))

# Let the user enter each value of the array
for i in range(arraySize):
    userArray.append(int(input("Element : ")))

# Reinitialize the array with given values
userArray = numpy.array(userArray)

# Print the initial array
print("\nThe initial array is :")
# print(numpyArray)    # Case 1 : Already declared array
print(userArray)      # Case 2 : User input array

# Replace all the values of the array that are lower than userThreshold value
# numpyArray[numpyArray < userThreshold] = 0    # Case 1 : Already declared array
userArray[userArray < userThreshold] = 0        # Case 2 : User input array

# Finally, we display the results
print("\nThe final array is :")
# print(numpyArray)             # Case 1 : Already declared array
print(userArray)                # Case 2 : User input array
##############################################


##############################################
# Exercise 2.2
print("\nExercise 2.2 :")

'''not finished'''
def show_in_digi(input_integer: int) -> None:

    # Firstly, we get the list of all the numbers in the integer, in the right order
    integersList = []

    while input_integer > 0:
        if input_integer < 10:
            integersList.insert(0, input_integer)
            break

        else:
            integersList.insert(0, input_integer % 10)
            input_integer = input_integer // 10
##############################################


##############################################
# Exercise 3
print("\nExercise 3 :")

# Display the dataset
dataset = pd.read_csv('california_housing_test.csv')
print(dataset)

# 3.1 : Check what dataset.describe() does
print("\nDescribe of the dataset :")
dataset.describe()

# 3.2 : Drop 1st and last rows
droppedDataset = dataset.drop([0, 2999])
print("\n")
print(droppedDataset)

# 3.3 : households column mean value
meanHouseholdsColumn = dataset["households"].mean()
print("\nHouseholds column mean value is :", meanHouseholdsColumn)

# 3.4 households column plotting
dataset["households"].plot()
xlabel = 'ID'
ylabel = 'Number of households'
pyplot.show()

# 3.5 : Check if there are any NaN values in the dataset and replace them with dataset mean value
dataset.isnull().values.any()
pd.dataset.mean()
datasetWithoutNaN = dataset.fillna(dataset.mean())
print("\nDataset without NaN values :")
print(datasetWithoutNaN)
##############################################