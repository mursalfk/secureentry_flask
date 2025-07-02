# IMPLEMENT THE FOLLOWING STEPS FOR FACE VERIFICATION
    1. need to create a db with embeddings of the persons' face that we want to verify
    2. then with the given image, we first get the embeddings of this new image
    3. we compare these two embeddings
        3.1 -> sample code uses cosine, we can change it to eucledian distance

in the [sample code][sample-code.py] database is created during runtime, we dont want that, we create it once
and when a new face is added, we update the db.

for the db we can use a simple json structure
like 
```
[
 {
    person_name: 'Person 1',
    embeddings: whatever we get from the mtcnn function
 }   
]
```

we load this database when comparing with the person and then compare
