# Put RouteWise on GitHub

Create a new repository named `routewise` on your GitHub account. Use this description:

> A fine-tuned support ticket router with human review, incident grouping, and a recorded demo.

Choose Public if you want evaluators to open the link without an invitation. If you keep
it Private, invite the evaluators before submitting the link.

Download and extract `RouteWise_submission.zip`. Open the extracted RouteWise folder.
On the new repository page, choose **uploading an existing file**, then drag the contents
of that folder into the upload area. Upload the files and folders themselves, not just
the ZIP. README.md should be at the top level of the repository.

Use this first commit message:

> Add RouteWise desk, recorded demo, and evaluation results

The code package includes the CPU replay and saved results. The original 127.4 MB model
backup is separate. Attach it to a GitHub Release if you want evaluators to restore the
exact model. Do not upload it through the ordinary repository file-upload page.
The live notebook checks the original ZIP fingerprint, so attach that original file.

After recording the Loom video, add its real link near the top of README.md. Use the link
text “Watch the three-minute walkthrough.” The recording should say that notebook 03 replays
saved GPU scores and that new-ticket inference is in notebook 01.

Before submitting, open the repository link in a signed-out window and check that the README,
demo notebook, screenshot, and video are accessible. Submit the repository URL through the
assignment form.
