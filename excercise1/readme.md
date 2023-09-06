Please implement simple web page with server:

- Imagine this application is used by multiple users
- Use plotly dash for implementation(https://plotly.com/examples/)
- application has simple but reasonable visual design (use boostrap to organize or pure CSS etc. ...)

Page should contain:
1. graph with employee data
    - please use pandas dataframe to load and filter excel/csv from provided /assets/data
    - show incidence matrix for which user is from which city where each point has different colour based on bonus field

2. input field with save button, each entry into this fields adds text as new row to a table
 - use plotly dash callbacks to implement this

3. selectbox
 - has list of 3 images
 - based on selection shows selected image from provided /assets/image

4. image click
 - when clicked into image, there is textfield that shows click coordinates on page
 - also print() this values in backend

BONUS:
 - on page, have element that that plays those 3 images as video
 - this does not have to use plotly callbacks but needs to be part of page
 - have play/pause buttons that stops or resume the video

Please spend no more than 2 hours on this*
Individual points are sorted according to their respective priorities
Please do as much as possible within those 2 hours
For what you did not make, please write main points on what you see as most significant challenges you expect to overcome if you had more time nad tryied to finish them also.

*You can go over "time budget" on BONUS item if you are willing to