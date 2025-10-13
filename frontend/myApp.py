from flask import Flask, render_template, request, flash 
import requests

app = Flask(__name__)


# The URL for your backend API endpoints
BACKEND_GET_REVIEW_IMAGES_URL = "http://localhost:8000/review-lookedup-images/"
BACKEND_SET_REVIEWED_URL = "http://localhost:8000/set-image-as-reviewed/"
BACKEND_RELABEL_URL = "http://localhost:8000/relabel-image/"


BACKEND_SINGLE_IMAGE_CLASSIFIER_URL = "http://localhost:8000/classify-image-url/"
BACKEND_MULTI_IMAGE_CLASSIFIER_URL = "http://localhost:8000/classify-image-urls/"
BACKEND_VIEW_JOB_URL = "http://localhost:8000/get_job/"


# CSRF security
app.config['SECRET_KEY'] = 'topSecretKey'

@app.route('/')
def index():
    title = "Image Classifier App"
    return render_template("index.html", title=title)

# This single route now handles both single and multiple URL submissions
@app.route('/classify/', methods=['GET', 'POST'])
def classify():
    title = "Classify Image(s)"

    if request.method == 'POST':
        # Check which form was submitted by looking at the submit button's 'name' attribute
        action = request.form.get('action')
        
        # --- Handle Single Image Classification ---
        if action == 'single':
            flash("Submitted single image URL for classification!")
            image_url = request.form.get('image_url')
            if not image_url:
                return render_template("classify.html", title=title, single_error="Please provide an image URL.")
            
            try:
                payload = {'image_url': image_url}
                response = requests.post(BACKEND_SINGLE_IMAGE_CLASSIFIER_URL, json=payload)
                response.raise_for_status()
                result = response.json()
                prediction = result.get('predicted_class', 'No prediction found.')
                confidence_level = result.get('confidence_level', 'No prediction found.')
                
                try:
                    confidence_level = float(confidence_level)
                except ValueError:
                    pass
                

                return render_template(
                    "classify.html", 
                    title="Classification Result", 
                    prediction=prediction.replace("_", " ").title(),
                    image_url=image_url,
                    confidence_level = round(confidence_level,2)
                )
            except requests.exceptions.RequestException as e:
                error = f"Could not connect to the backend at {BACKEND_SINGLE_IMAGE_CLASSIFIER_URL}. (Error: {e})"
                return render_template("classify.html", title=title, single_error=error)
            except Exception as e:
                return render_template("classify.html", title=title, single_error=f"An error occurred: {e}")

        # --- Handle Multiple Image Classification ---
        elif action == 'batch':
            urls_text = request.form.get('image_urls')
            if not urls_text:
                return render_template("classify.html", title=title, batch_error="Please provide at least one image URL.")
            
            url_list = [url.strip() for url in urls_text.splitlines() if url.strip()]
            if not url_list:
                return render_template("classify.html", title=title, batch_error="Please provide at least one valid image URL.")

            try:
                payload = {'urls': url_list}
                response = requests.post(BACKEND_MULTI_IMAGE_CLASSIFIER_URL, json=payload)
                response.raise_for_status()
                job_id = response.json().get('job_id', "None")
                flash(f"Submitted multiple image URL for classification! view job with job id: {job_id}")
                
                return render_template(
                    "classify.html", 
                    title="Batch Classification Results", 
                    job_id=job_id
                )
            except requests.exceptions.RequestException as e:
                error = f"Could not connect to the backend at {BACKEND_MULTI_IMAGE_CLASSIFIER_URL}. (Error: {e})"
                return render_template("classify.html", title=title, batch_error=error)
            except Exception as e:
                return render_template("classify.html", title=title, batch_error=f"An error occurred: {e}")

    # For a GET request, just show the initial page
    return render_template("classify.html", title=title)


# NEW ROUTE for viewing job results
@app.route('/view-job/', methods=['GET', 'POST'])
def view_job():
    title = "View Classification Job"
    if request.method == 'POST':
        job_id = request.form.get('job_id')
        if not job_id:
            return render_template("viewJob.html", title=title, error="Please provide a Job ID.")
        
        try:
            # Construct the full URL and make a GET request
            response = requests.get(f"{BACKEND_VIEW_JOB_URL}{job_id}")
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            
            data = response.json()
            job_status = data.get('status')
            results = data.get('result', [])

            # Pre-process results to format prediction and confidence
            # formatted_results = []
            # if results:
            #     for res in results:
            #         confidence = res.get('confidence_level', 0)
            #         try:
            #             # Convert to percentage and round it
            #             confidence = round(float(confidence) * 100, 2)
            #         except (ValueError, TypeError):
            #             confidence = "N/A"
                    
            #         formatted_results.append({
            #             'image_url': res.get('image_url', ''),
            #             'predicted_class': res.get('predicted_class', 'N/A').replace("_", " ").title(),
            #             'confidence_level': confidence
            #         })

            return render_template(
                "viewJob.html", 
                title=f"Results for Job {job_id}",
                job_id=job_id,
                job_status=job_status,
                results=results
            )

        except requests.exceptions.HTTPError as e:
            # Handle cases like 404 Not Found specifically
            error = f"Job ID '{job_id}' not found. Please check the ID and try again."
            return render_template("viewJob.html", title=title, error=error)
        except requests.exceptions.RequestException as e:
            error = f"Could not connect to the backend. (Error: {e})"
            return render_template("viewJob.html", title=title, error=error)
        except Exception as e:
            error = f"An unexpected error occurred: {e}"
            return render_template("viewJob.html", title=title, error=error)
            
    # For a GET request, just show the initial page
    return render_template("viewJob.html", title=title)


@app.route('/review/')
def review_images():
    title = "Review Images"
    try:
        response = requests.get(BACKEND_GET_REVIEW_IMAGES_URL)
        response.raise_for_status()
        images_data = response.json()

        # Pre-process data for easier use in the template
        # formatted_images = []
        # for img in images_data:
        #     confidence = img.get('confidence_level', '0')
        #     try:
        #         # Convert confidence to a rounded percentage
        #         confidence = round(float(confidence) * 100, 2)
        #     except (ValueError, TypeError):
        #         confidence = "N/A"
            
            # img['confidence_level'] = confidence
            # Use local_url if available, otherwise fall back to the public url
            # img['display_url'] = img.get('local_url') or img.get('url')
            # formatted_images.append(img)

        return render_template(
            "reviewImages.html", 
            title=title, 
            images=images_data,
            backend_set_reviewed_url=BACKEND_SET_REVIEWED_URL,
            backend_relabel_url=BACKEND_RELABEL_URL
        )

    except requests.exceptions.RequestException as e:
        error = f"Could not connect to the backend to fetch images. (Error: {e})"
        return render_template("reviewImages.html", title=title, error=error)
    except Exception as e:
        error = f"An unexpected error occurred: {e}"
        return render_template("reviewImages.html", title=title, error=error)

if __name__ == '__main__':
    app.run(port=5000, debug=True)

