# Using a table with data from CSVs

Please implement simple web page with server:

- Imagine this application is used by multiple users
- Use plotly dash for implementation(https://plotly.com/examples/)
- application has simple but reasonable visual design (use boostrap to organize or pure CSS etc. ...)


Page should contain:
- 1 input field which will query the the data source and filter the data
- 1 editable Dash table in tied to the input field

Table should:
- be filterable and searchable
- have a function to delete a row 

Functionality: 
- upon choosing one row in the Dash table a dialog should open which asks whether the user is sure he wants to delete the row 
- A call should be made to delete the row in the data source and save it to a different backup file
