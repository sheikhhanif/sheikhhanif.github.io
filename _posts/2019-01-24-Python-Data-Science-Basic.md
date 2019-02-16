---
title: "Python for Data Science - Part 1"
header:
  teaser: /assets/images/python/logo.png
date: 2019-01-24
tags: [python]
excerpt: "Along the way you will be learning about following topics.
- How to print, comment
- Python String
- Python list
- Python packages
- Numpy"
---


This is the first part of Python for Data Science. I will provide the most basics of Python to get started your data
science journey. Along the way you will be learning about following topics.
- How to print, comment
- Python String
- Python list
- Python packages
- Numpy

### Python Basics
Python is a versatile language. You can use it to 
perform pretty much everything you want. From analyzing and building ML model
to building efficient web application can be done easily.
Ok, that's enough! Let's get into the business


### Printing and conmmenting
If you want to print a statement you just need to call python pirnt()
function. Let's say I wanto print 5. I just need to pass 5 inside print() function as follows


```python
print(5)
```

    5
    

We can perform mathematical calcualtion in python natural way.
Let's say, I want to add two numbers. I can just do it as follows:-


```python
addNumber = 5 + 4

print(addNumber)
```

    9
    

Hold on! You might be wondering what is that 'addNumber' comes from. You can refer to it as 
variable. In Python, a variable allows you to refer to a value with a name. To create a variable use =, like this example: x = 10.

Another way to think is that Python is an object-oriented programming language, and in Python everything is an object. In our case 
'addNumber' refer to an object which is (5 + 4).
You can think yourself as an object where you have some value, 
attribute and so on. In future post you can learn more about object.

Now, you should be able to do other mathematical operation. Let's move on.

How about you want to comment something in your code.
You can use comment as follows: -
Single Line Commment: 
- this is single line comment
- """ This is for multiple
line comment"""
Bear in mind that """ """ is docstring where you can do multiple
lines of comment within it.


```python
# I'm comment. I won't be executed
print(10)
```

    10
    

Let's solve a very simple real life problem. Imagine you went 
to market. You have bought 5 pens 1 dollar each, a excercise book with 8.5 dollars.
Let's calculate the total cost.


```python
# cost of items
pen_cost = 5 * 1
book_cost = 8.5

# total cost
total_cost = pen_cost + book_cost
print("Total spent", total_cost)
```

    Total spent 13.5
    

### Type
There are different types of objects in python. Some
of them are integer, string, boolean.
To evaluate the type of object we can use Python 
type() fucntion.


```python
print("type of 'pen_cost' oject is: ", type(pen_cost))
```

    type of 'pen_cost' oject is:  <class 'int'>
    


```python
my_name = "Hanif"
print("type of 'my_name' oject is: ", type(my_name))
```

    type of 'my_name' oject is:  <class 'str'>
    

You can evaluate any type of object using this funcion. Try yourseft by 
creating more variables

### String
We know that we can do mathematical operation on numbers
more precisely on 'int', 'float' objects. How about you want to 
add two string.
You can do as follows: 



```python
first_name = "Sheikh"
last_name = "Hanif"
full_name = first_name + last_name
print("My full name is: ", full_name)
```

    My full name is:  SheikhHanif
    

In above code block I am declaring three variables.
Where I am referring two 'str' objects with first two variables
And then concatenating these two objects and store into 
third variables which is 'full_name'.

Every object has it's own attributes and methods in it.
'str' object has bunch of methods in it.
You can see what methods and attributes available in 
an object as follows:


```python
dir(str)
```




    ['__add__',
     '__class__',
     '__contains__',
     '__delattr__',
     '__dir__',
     '__doc__',
     '__eq__',
     '__format__',
     '__ge__',
     '__getattribute__',
     '__getitem__',
     '__getnewargs__',
     '__gt__',
     '__hash__',
     
     'find',
     'format',
     'format_map',
     'index',
     'isalnum',
     'isalpha',
     'isascii',
     'isdecimal',
     ..............
     'swapcase',
     'title',
     'translate',
     'upper',
     'zfill']



As you can see there are a lot of methods and attribtes of string object.
Let's try few of them.


```python
# Uppercase
full_name = full_name.upper()
print(full_name)

# counting specific alplhabel in string
count_H = full_name.count('H')
print(count_H)
```

    SHEIKHHANIF
    3
    

### List
A list can contain any Python type. Although it's not really common, a list can also contain a mix of Python types including strings, floats, booleans, etc.


```python
# creating list
numbers = [10, 20, 11, 99, 40]

# print list
print(numbers)
```

    [10, 20, 11, 99, 40]
    

We can access the items / objects / values in list
by index. In programming index start with 0. You can print
the last number as follows.


```python
# print the last number from list
print(numbers[4])
```

    40
    

We can even slice a list. You can print particular items
you want as follows:


```python
# print first 3 items
print(numbers[:3])
```

    [10, 20, 11]
    

Replacing list elements is pretty easy. Simply subset the list and assign new values to the subset. You can select single elements or you can change entire list slices at once.


```python
# replacing last element from the list
numbers[4] = 100

# printing the list
print(numbers)
```

    [10, 20, 11, 99, 100]
    

You can add element in a list and also delete an element
from it as follows


```python
# adding element
numbers.append(150)

# delete an element 
del(numbers[0])
```

Now let's look into our list after deleting the first
item and adding 150 in it.


```python
print(numbers)
```

    [20, 11, 99, 100, 150]
    

### Packages
Pythons has large collection of packages. It's enriched
with a lot of scientific and data science packages. In order to 
work with these packages you need to import them. Importing
packages is pretty much easy in python. Like asking your mom to give you
food. Here is how : - 


```python
# importing math package
import math
```

Pretty simple rite!! How about you want to import specific
method / function from a package. You can do so as
follows - 


```python
# importing pi from math modules
from math import pi
```


```python
# let's utilize our package
# as we already imported it.

# calculating area of circle
radius = 5
area = pi * radius**2 

# printing area
print("The area of circle with radius 5 is: ", area)
```

    The area of circle with radius 5 is:  78.53981633974483
    

### Numpy
Numpy is one of the python scientific package which 
you gonna love. Because it's super fast, easy and best tool 
to work with n-dimensional arrays. OPPS!!! 'arrays'!!
I didn't mention it before right. Remember our list called
'numbers'. It's an one dimensional array. Now we will work
with numpy array. Let's go.

As I mentioned earlier, if we want to use any package we need to 
import them first. Let's do it!


```python
# importing numpy
import numpy as np
```

I import the 'numpy' package and give it a name 'np'
so that I can refer it as 'np' later on. You can give it 
any name you want. But by convention everyone use
'np'. So if you use np everone will understand it as
numpy. Enough talk! Let explore...


```python
# creating numpy array
nums = np.array([39, 12, 43, 11, 90])

print(nums)
```

    [39 12 43 11 90]
    


```python
# let's add 1 in each element of it
# Python broadcasting. Will tank next episode
nums = nums + 1
print(nums)
```

    [40 13 44 12 91]
    

We can now do a lot of operation in 'nums'. Let's check what methods and attributes are available in numpy. We will try out few of them


```python
# getting available numpy attributes and methods
dir(np)
```




    ['ALLOW_THREADS',
     'AxisError',
     'BUFSIZE',
     'CLIP',
     'ComplexWarning',
     'DataSource',
     ------------
     'where',
     'who',
     'zeros',
     'zeros_like']



We can create numpy arrays with zeros and ones in it
as follows.


```python
# array with zeros
z_arr = np.zeros(10)
print(z_arr)
```

    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    


```python
# array with ones
o_arr = np.zeros(10)
print(o_arr)
```

    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    

I can create 2d numpy array as well. At the same time,
I can resize an array however I want.


```python
# creating an 2d array
arr = np.array([[1,2], [3,4], [5,6]])
print(arr)
```

    [[1 2]
     [3 4]
     [5 6]]
    


```python
# to get the shape
print(arr.shape)
```

    (3, 2)
    


```python
# to reshape
arr = arr.reshape(2,3)
print(arr)
```

    [[1 2 3]
     [4 5 6]]
    

To access element in multi-dimensional array, first 
you need the specify the row number and then column.
If I want to access 5 from 'arr'. I can access it as follows: - 


```python
print(arr[1,1])
```

    5
    


```python
# printing entire second row
print(arr[1,:])
```

    [4 5 6]
    

You can do a lot more with numpy. In the exciting world of data science, there are plenty of tools and package to work with. That's 
all for introduction to python for data science series. Check out rest of my blog for more resources.
