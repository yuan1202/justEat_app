# Setup & Requirements

This solution assumes you have the following installed:
* Python 3.6 or later
* Docker

To run the solution, please first build the docker container:

`docker build -t jetmlett .`

And then it can be run via:

`docker run -d -p 5000:5000 jetmlett`

Once the container is running in the background, it is listening on port 5000.

The solution has two endpoints: `/infer` for single query; and `/batch` for batched queries.

As specified the two endpoints should be accessed in the following format:

infer:

`
POST http://localhost:5000/infer
{ "name": "mike", "vector": [0,1,0] }
`

batch:

`
POST http://localhost:5000/batch
[{ "name": "mike", "vector": [0,1,0] }, ... etc. ]
`

# Solution description

1. In Dockerfile, a more lightweight python:3.9.0-slim-buster image is used instead of python:3.9.0-buster. The resulting image size is reduced from 982MB to 212MB, improving the scalability of the app. (\#performance \#scalability)

2. A Recommender class is implemented and instantiate. It loads the product embedding at instantiation and hence only does it once. This significantly reduces the response time compared to the original implementation where the data has to be loaded at every request. (\#performance \#scalability)

3. The use of the Recommender class assumes that the model, i.e. the product embedding, fits in memory entirely. In reality, this may not be the case and a different approach may be needed. Besides, using some existing datastore solution, such as Redis, might help to further improve the scalability of the application. (\#performance \#scalability)

4. The Recommender also implements the dot product computation in a vectorised manner for both single query and batch query. This will also help to reduce response time considerably. Combined with pre-loading product vector data, it reduces the response time from ~0.12 second down to ~0.01 second. (\#performance \#scalability)

5. The Flask application is deployed with Gunicorn package, which allows concurrent and asynchronous processing of the requests. It also manages and instances of the application automatically and restarts them when necessary, greatly improving the reliability. In the test case with 1000 concurrent single requests (Gunicorn with 4 workers), it can maintain an average response time of ~0.01 second. (\#performance \#scalability \#reliability)

6. An example config.py file is included to demonstrate the common practice in configuring the web application, which will make the app configuration task easier and more reliable. (\#reliability)

7. In its basic form, recommendations are generated by finding the dot product between product embeddings and user embedding. Depending on how the embeddings are generated, it is potentially a better measurement then simple cosine similarity, i.e. it captures the magnitude of the "similarity" which could represent the intensity of the user's favour. However, intuitively it may be sensible to firstly group the products into categories such as main course, side dishes, and desserts, etc, and then perform dot product within each group to generate recommendation against each category. Some basic data explorations have been performed on the product vectors using PCA and t-SNE. Unfortunately, no clear grouping has been identified. Due to time constraint, no further action is taken on this topic. (\#makesense)

8. In a complete solution where clear specifications are provided, unit tests should be implemented to properly capture edge cases. For this technical test, edge case tests have been implemented in jupyter notebook. Two main types of edges cases are handled within the app: formatting error in the serialised request data; and nan/inf cases in the vector data. The prior is handled at the endpoint call with try/except, i.e. if an error is encountered, a response with 400 status code is returned. The latter is handled within the Recommender instance, where if nan/inf is encountered, None will be returned as recommendation results. (\#edgecase)

9. All the tests are provided in the tests.ipynb notebook. It concludes basic query and batch query test, stress tests, as well as edge case tests. (\#edgecase \#tests)

# Further work

1. Implement proper unit tests, using either unittest or pytest, to capture edge cases.

2. Some means of authentication should be also included to protect the app from unnecessary request.

3. As mentioned in point \#3 of the previous section, proper datastore solution can be used to improve reliability and scalability.

4. As mentioned in point \#7 of the previous section, more tailored recommendation may be needed to improve user experience.