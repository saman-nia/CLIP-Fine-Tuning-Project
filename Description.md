# Assessment ML-Engineer

This folder contains a Jupyter notebook "train.ipynb" as well as two subfolders, "data" and "models". The data is art images from wikipedia alongside descriptions in a JSON file. Let's pretend these are customer data (e.g. Images that have been made with the phone on the construction size with our app, together with ticket descriptions).

The notebook is a simple version of something you or someone on the team might end up after experimenting. In this case, it finetunes a visual encoder of a CLIP-like model on custom data. Of course, the data here is much too litte, so it overfits massively, but it should not be hard to imagine a real-world use case with enough data. The end result is a model that clients can use for generating multi-modal embeddings of images and texts, e.g. for cosine-similarity search in a vector database (not part of the assessment).

What is the task? Simply stated: Take this notebook and turn it into something production-ready. The end result should be some runnable code, together with some text that explains what's missing to turn your code into a full-fledged production-ready deployment.

It is assumed that the training data grows regularly (because our customers can't stop taking images), so the code is assumed to run every month to produce a model that takes the new data into account. The data would be stored in a S3 bucket. In your code, just use the local storage and indicate with comments or pseudo-code where the calls so S3 would happen.

Your result should at the least be a runnable container. Things you should touch upon either in code or in a written explanation include (but is not limiter to):

- documentation
- - including operational – no need to write the full docs, write what you would cover
- configuration
- logging
- monitoring
- runtime environment
- - you may assume an AWS cloud environment or something comparable
- model versioning
- model storage & deployment
- notifications and warnings

The end result should be enough for someone to take your code and instructions and turn in into a deployment that allows clients to get a new model every mongth. Assume the client knows how to use CLIP-like models (no need to document that). Rather than go into too great detail with every point, be brief and cover the all of what you think is necessary. You may use LLMs and other tools as you wish, but you should be able to argue for the decisions made.

There is no one correct solution. We're looking for a solution that is complete and robust while avoiding unecessary complexity.

If there are any open questions, don't hesitate to ask!