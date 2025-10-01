### Image Classifier App
A program that allows a user to classify images by submitting url and get a response of the job ID.
The job ID is used to index the results of the classsifier. Once the job is complete you can request for the result.


 ## Project Structure

A brief overview of the key directories in this project:
-   **/images**: Contains all images used for training the model, both environment (`/images/environment`) and studio (`/images/studio`).
-   **/furniture_dataset**: Contains images from (`/images`) stored in 70|15|15 train|val|test split.
-   **/ltd-flooring-surfaces-materials-image-csv**: Contains csv files of images from ltd platform.
-   **/mobilenet_v3_small_model**: Contains the transfer-learning CNN model that used mobileNetV3Small.
-   **/mobilenet_v3_small_model-old**: Contains the transfer-learning CNN model that used mobileNetV3Small that was trained on a smaller dataset.
-   **/scraper**: Contains script that allows you to scrape images from arper media library using playwright.

