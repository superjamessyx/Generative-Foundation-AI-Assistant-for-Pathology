# PDF extracter

This tool is used for extracting the specific figures with their corresponding captions / descriptions from the e-books (pdf format). The main idea is to convert the book file for **.pdf** format to **.html**, then parsing HTML file to obtain the specifical contents by the corresponding *\<sign\>*.
Hence you need the following requirements,

+ [pandoc](https://github.com/jgm/pandoc)
+ Python packages:
  + [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/): HTML parsing
  + [tqdm](https://github.com/tqdm/tqdm): Visualize execution 

---
## Quick start
+ put the **batch_process.py** in YOUR BOOKS FOLDER
+ run the python script **batch_process.py**
+ this code will generate the folders for each e-book (.pdf) as following file structure  
├── **Book name**  
│    ├── book name.pdf  
│    ├── book name.html  
│    ├── book name.doc  
│    ├── book name.json  
│    ├── **mdeia convert**  
│    │        ├───Fig.1.jpg  
......


 