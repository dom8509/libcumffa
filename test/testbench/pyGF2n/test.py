
def my_print_decorator(range):
    def my_print_wrapper(fun):
        x = range
        for i in x:
            fun(i)
    return my_print_wrapper

@my_print_decorator(range(0,10))
def my_print(x):
    print "x = ", x


