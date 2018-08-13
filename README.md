# BM25f
This python code implements the BM25f algorithm.
Input CSV file has been provided along with output CSVs generated according to the search query.

Assumptions:
1)Input CSV file is in the form
___________________ 
| title | body     |
|_______|__________|

here each row represents one article

2)Output CSV is in the form of
___________________ ______________
| title | body     |  relevance   |
|_______|__________|______________|

3)For a given Query Q, containing keywords q1, q2,..., qn, the BM25 score is 
score(document , Q ) = sum_for_qi( idf(qi) * ( f(qi , D)*(k+1) ) / ( f(qi , D) + k*(1 -b + b*|D|/avgdl) ))  where i = 1 to n 

4)For BM25f , structured document is converted into unstructured document using:
unstructured_doc = title_weight*title + body_weight*body 
 

For testing:

Test Case 1)enter query:
            eyes
Test Case 2)enter query:
            eyes and  vision
Test Case 3)enter query:
            ears            

 
 
