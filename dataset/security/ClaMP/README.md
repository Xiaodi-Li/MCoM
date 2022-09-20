# ClaMP (Classification of Malware with PE headers)

A Malware classifier dataset built with header fields’ values of Portable Executable files

<p align="center">
<img width="700px" src="ClaMP-process.jpg" alt="ClaMP process diagram"/>
</p>

# Dataset files
1. ClaMP_Integrated-5210.arff
	- Total samples	: 5210 (Malware (2722) + Benign(2488))
	- Features (69)	: Raw Features (54) + Derived Features(15)
2. ClaMP_Raw-5184.arff
	- Total samples	: 5184 (Malware (2683)+ Benign(2501))
	- Features (55)	: Raw Features(55)

3. ClaMP_Integrated-5184.csv
	- Total samples	: 5184 (Malware () + Benign())
	- Features (69)	: Raw Features (54) + Derived Features(15)
4. ClaMP_Raw-5184.csv
	- Total samples	: 5184 (Malware ()+ Benign())
	- Features (55)	: Raw Features(55)

# Scripts files
  Note: For many purposes the [pefile](https://github.com/erocarrera/pefile) have been used.
    I want to thank Ero Carrera for creating the great tool **pefile**.

1. **move_duplicate_files.py**
      This python script will move the duplicates files based on their MD5 and will give file information report as CSV file. ( Look into script header for detail working)

2. **calculate_size_and_entropy.py**

      This python script will calculate size and entropy of all files present in given directory and will write these information with file name to a .csv file.

3. **benign-labeling.py**
    This script will take a csv file with MD5 hash as input and it will read all MD5 and will fetch the VirusTotal report on each MD5 and after receiving and parsing the report
    will write them to a CSV file path/report.csv.  The CSV file header will have fields as:

      - *ID, fileName, MD5hash, Total, Positive,*
      - *type-TrendMicro, type-F-secure, Scan-Date*

4. **get_all_md5.py**       

      This is very simple script make use of python MD5 function to calculate MD5 hash of all files in given directory. filename and their MD5 written to a csv file to later use. This file can be use with benign-abeling.py script to get VirusTotal report.

5. **get_PE_file_info.py**  
      This script will give file information related to company and version.
      - File version
      - ProductVersion
      - ProductName
      - CompanyName
      - etc..
6. **integrated_features_extraction.py**    

      This is one of main file for creating dataset, it will extract integrated features
       of given samples. User have to supply malware and benign samples path in two
        different run and supplying the class label accordingly. Currently, user
        have to merge extracted   features manually for both classes to create overall dataset.

7. **raw_features_extraction.py**    

      This is another important script which extract all the values from all three main
      PE headers. DOS_Header, FILE_HEADER AND OPTIONAL_HEADER. If any exception happens      
      then the values will be assigned as zero for that header. Say, many
      PE files don't have DOS_Header then all the header will be assigned '0'.

      - IMAGE_DOS_HEADER (19)
        - "e_magic", "e_cblp", "e_cp","e_crlc","e_cparhdr",
        - "e_minalloc","e_maxalloc","e_ss","e_sp",
        - "e_csum","e_ip","e_cs","e_lfarlc","e_ovno","e_res",
        - "e_oemid","e_oeminfo","e_res2","e_lfanew"

      - FILE_HEADER (7)
        - "Machine","NumberOfSections","CreationYear","PointerToSymbolTable",
        - "NumberOfSymbols","SizeOfOptionalHeader","Characteristics"

      - OPTIONAL_HEADER   (29)
        - "Magic","MajorLinkerVersion","MinorLinkerVersion","SizeOfCode","SizeOfInitializedData",
        - "SizeOfUninitializedData","AddressOfEntryPoint",
        - "BaseOfCode","BaseOfData","ImageBase","SectionAlignment","FileAlignment",
        - "MajorOperatingSystemVersion","MinorOperatingSystemVersion",
        - "MajorImageVersion",  "MinorImageVersion",  "MajorSubsystemVersion",
        - "MinorSubsystemVersion",  "SizeOfImage",  "SizeOfHeaders",  "CheckSum",
        - "Subsystem",  "DllCharacteristics",  "SizeOfStackReserve",  "SizeOfStackCommit",
        - "SizeOfHeapReserve",  "SizeOfHeapCommit",  "LoaderFlags",  "NumberOfRvaAndSizes"

8. **select_malware_sample_as_VT_report.py**      

        This is kind of supportive script to automate the process of selecting samples
         from initial samples according to the detection result of top 10 Anti-virus
         engines at VirusTotal. Suppose, we only want to keep those sample as malware
         for which out 9 out
        of 10 AV have given malware flag. By changing threshold value we can have
        different group of samples. Script will move the samples to a new folder.

9. **malware-labeling.py**  

      This script will take a csv file with MD5 hash as input and it will read all MD5 and
      will fetch the VirusTotal report on each MD5 and after receiving and parsing the report,
      will write them to a CSV file path/report.csv.  The CSV file header will have fields as

    - *"MD5hash", "Total", "Positive",*
    - *"TrendMicro", "F-Secure", "McAfee", "Symantec", "Avast", "Kaspersky",*
    - *"BitDefender", "Sophos", "GData", "Panda", "Qihoo-360",  "Scan-Date"*

10. **select_benign_sample_as_VT_report.py**   

        This is kind of supportive script to automate the process of selecting samples from
        initial samples according to the detection result of  VirusTotal. Suppose,
        we only want to keep those sample as benign for which none of the AVs have
        given malware flag. This will also move samples for which past analysis
        result is not available at VirusTotal (Because we are not submitting
        sample instead getting result by MD5).Script will move  the samples
        to a new folder according to /notbenign and /noreport.

11. **filetype.sh**

        A Simple shell script to use Linux's file command to test file type of all files
        in given directory. Later the output text file can be parse to create some report.

12. **CSVToARFF.py**

        This python script convert CSV file to ARFF (default WEKA file). For detail see the
        header part of script. It is an updated code on a earlier code on github.
        Please see header for detail.

13.  **peid.yara**

        This the PEiD's packers signature converted as yara rules.
        Used in integrated_features_extraction. Please refer feature_extraction file.



# Raw samples metadata information

1. **Clean_md5_2917.csv**

    This file have filename,MD5 hash and size for all clean samples (2917) collected for experiment.

2. **Malware_md5_2917.csv**    

    This file have filename,MD5 hash and size for all malware samples (2917) collected for experiment.

3. **Clean_md5_without_dup_2873.csv**

     This file have filename,MD5 hash and size without any duplicate clean samples (2873).

4. **Malware_md5_without_dup_2873.csv**   

     This file have filename,MD5 hash and size without any duplicate clean samples (2873).
5. **Malware-2722_hash_size_entropy.csv**  

    This file have filename (Hash) , file-size in bytes, and Entropy of each malware sample a Total of 2722.

6. **Clean-2501_name_size_entropy.csv**  

    This file have filename, file-size in bytes, and Entropy of each clean sample, which where collected after fresh installed Windows OS (XP and Windows7) a Total of 2501.

7. **Clean_VT_report-2873.csv**    
    This file have Virus Total report of all clean samples (without duplicate, 2873). File                contains information like:
    - *ID, fileName, MD5Hash, Total, Positive, Type-TrendMicro, Type-F-secure, Scan-Date.*

8.  **Clean_NOT_PE_6.txt**  

      This file have list of clean files which are not Portable Executable (PE) file format.

9.  **Malware_VT_report_without_Zipped_3817.csv**  

    This file have Virus Total report of all malware samples (with some zipped that is not used in analysis, 3817). File contains information like,

    - *MD5hash, Total, Positive, TrendMicro, F-Secure, McAfee, Symantec, Avast, Kaspersky,  BitDefender, Sophos, GData, Panda, Qihoo-360, Scan-Date*

### Features with their description
    S.NO | Feature | Type | Description
    -----|-------- | -----| ------
    01 | e_magic | Integer | It is in the DOS_HEADER and it must be same for all.
