# MLEngineer Tech Test

Welcome to the Just Eat Takeaway Machine Learning Engineer Tech Test! (or JETMLETT!).

This is our brand new tech test for MLEngineering, and we're keen for feedback. If it feels too heavyweight/lightweight or you have thoughts how we can improve things going forward, please give us feedback! Thanks!

## Scope & Expectation

A lot of this tech-test is intentionally open-ended. You're welcome to spend as much time as you like on the tasks below.

If there's anything you'd like to implement but haven't got the time for, jot down some notes in the `answers.md` file which describes what you would have done.

### Output

In order to avoid bounced emails we would like you to submit your code by uploading the zipped repo to a shared Google Drive folder. In order to obtain the URL for this folder please supply your Gmail or Google-based email address to either your agent or the JUST EAT member of staff who assigned you the test.

Please make this a single zip file named {yourname}-{role-applied-for}.zip

## Contents

We have provided a flask api which returns recommended products based on the likes of a given user.
The recommendations are generated by comparing 300 dimension vectors between users and products from a word embedding.

The word embedding itself is not included, only the output data for products and the query data for users

## Setup & Requirements

This tech tests assumes you have the following installed:
* Python 3.6 or later
* Docker

To run the solution, first you need to build the docker container:

`docker build -t jetmlett .`

And then it can be run via:

`docker run -d -p 5000:5000 jetmlett`

This will spin up the container running in the background, listening on port 5000.

You can then send http request to `localhost:5000` as described below.

We have also provided a method of testing the `infer` endpoint, via `query.py`. If you want to use this, we recommend creating a new `virtual environment` and installing the contents of `requirements.txt` before running.

---

## Tasks

### Productionisation

The API contains one endpoint for inference that can be called as follows

```
POST http://localhost:5000/infer
{ "name": "mike", "vector": [0,1,0] }
```

We have provided test data for querying in `users.json`

Best product matches and worst matches are then returned.

The endpoint works fine, but might not be quite right for operation in Production.

*We would like you to productionise the `infer` endpoint and ensure it is performing as expected.*

This is very open ended, so feel free to timebox it to a couple of hours, tackling just the important bits!

Some things to consider:
* Is the endpoint performant?
    * Reliable?
    * Scalable?
* Do the results make sense?
* Are edge-cases handled?

### Batch Endpoint

Now that we have a fabulous `infer` endpoint, we now want to create a new endpoint for `batch inference`. This endpoint will take a list of users and provide results for all of them in one go.

```
POST http://localhost:5000/batch
[{ "name": "mike", "vector": [0,1,0] }, ... etc. ]
```

We have provided an example payload for this endpoint in `users_batch.json`

*We would like you to implement the `batch` endpoint*

Some things to consider:
* Is the endpoint performant?
    * Reliable?
    * Scalable?
* Are edge-cases handled?