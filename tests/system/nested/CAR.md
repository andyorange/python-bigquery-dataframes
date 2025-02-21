
# Not suggesting a framework, but a mixed approach of a "Handler" class and custom methods, packing the processing steps in logic blocks

Use ColumnManager including its config file to setup what is processed how: Sections/Methods for each logic block of preprocessing, such as Impute-Zero, Impute-const, Impute-median, OneHotEncode, CategoryMerge, which includes all the columns that will be processed in the respective way

Simple way of getting columns for action -> getColumns(Processing.T_Impute_Zero) (and here I WOULD love the MemberSelector to get away from using strings)
Use "Blocks" of preprocessing actions, so they can be tested and debugged easily. And helps organizing

def Drop(cm: ColumnManager)
    ...

def Impute(cm: ColumnManager):
    ... mean, zero, const, median
    Returns list of tuples that can be used in the ColumnTransformer, the names
       coinciding with their respective columnManager

def OneHotEncodeAndCatReduce(cm: ColumnManager):
    ...



* If we want to automate things: Create simple Function (or class or whatever) that creates a ColumnTransformer for each new step that does all the things in parallel mentioned in the step using the ColumnManager

AddStep(Drop, Impute)
AddStep(OneHotEncodeAndCatReduce)


* If we have a PreProcessing Handler Object we can even tell where to add the step to, to which processing (in that case, the above function would be for instance an objewct created by a factory or a class such as PreProcTrain, PreProcTest)

PreProcX.AddStep(...) and it would be similar to the base Class approach by Julian but with the "Actions" separated out.
This could be also packed into a Decorator instead, which is then part of a class that knows respective PreProcessing Paths ("Train", "Test")

@PreProc(["Train"])
AddStep(...)

@PreProc(["Train", "Scoring"])
AddStep(...)

Advantage: The same code for Train and Test and one call the respective path in one line only that is different

* Non SKLearn things such as complicated pandas stuff with apply and lambdas or column arithmetics can be addressed similarly, only perhaps outside the SKLean ColumnTransformer. Or one creates an SKLearn Custom Transformer, as in most cases that is anyway use cases specific
