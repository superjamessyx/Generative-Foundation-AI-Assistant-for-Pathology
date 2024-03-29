# Figure extracter

This tool is used for extracting the specific figures with their corresponding captions / descriptions from the e-books. The main idea is to convert the book file for **.doc** format to **.html** format, then parsing HTML file to obtain the specifical contents by the corresponding *\<sign\>*.

**Note:** If the e-book format is **.pdf**, you need to prepare a transfer tool (such as [Adobe Acrboat](https://acrobat.adobe.com/)) to convert the format from **.pdf** to **.docx**.

Hence you need the following requirements,

+ [pandoc](https://github.com/jgm/pandoc)
+ Python packages:
  + [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/): HTML parsing
  + [tqdm](https://github.com/tqdm/tqdm): Visualize execution process

---
## Quick start
+ put the **batch_process.py** in YOUR BOOKS FOLDER
+ run the python script **batch_process.py**
+ this code will generate the folders for each e-book as following file structure  
``````
├── **Book name**  
│    ├── book name.pdf  
│    ├── book name.html  
│    ├── book name.docx  
│    ├── book name.json  
│    ├── mdeia convert 
│    │    ├───Fig.1.jpg  
...
``````

~~this code will convert the **book name.pdf** to **book name.docx**~~, This code the **book name.docx**  to **book name.html**, which will be prased and extracted the figures and captions. The extracted results will be saved in dict as **book name.json**.  
**book name.json** records all figures' names (or replace with relative path) and their correspending captions.

 
