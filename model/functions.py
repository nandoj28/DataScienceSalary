import re

# This mapping is for condensing the country codes into regions. 
region_mapping = {
    'US': 'North America', 'CA': 'North America',
    'AR': 'Latin', 'BO': 'Latin', 'BR': 'Latin', 'CL': 'Latin', 'CO': 'Latin', 'CR': 'Latin', 'CU': 'Latin', 
    'DO': 'Latin', 'EC': 'Latin', 'GT': 'Latin', 'HN': 'Latin', 'MX': 'Latin', 'NI': 'Latin', 'PA': 'Latin', 
    'PE': 'Latin', 'PR': 'Latin', 'PY': 'Latin', 'SV': 'Latin', 'UY': 'Latin', 'VE': 'Latin', 'BS': 'Latin',
    'AT': 'Western Europe', 'BE': 'Western Europe', 'CH': 'Western Europe', 'DE': 'Western Europe', 'DK': 'Western Europe', 
    'ES': 'Western Europe', 'FI': 'Western Europe', 'FR': 'Western Europe', 'GB': 'Western Europe', 'IE': 'Western Europe', 
    'IS': 'Western Europe', 'IT': 'Western Europe', 'LU': 'Western Europe', 'NL': 'Western Europe', 'NO': 'Western Europe', 
    'PT': 'Western Europe', 'SE': 'Western Europe', 'AD': 'Western Europe', 'JE': 'Western Europe', 'MT': 'Western Europe', 'GI': "Western Europe",
    'BG': 'Eastern Europe', 'CZ': 'Eastern Europe', 'EE': 'Eastern Europe', 'HU': 'Eastern Europe', 'LV': 'Eastern Europe', 
    'LT': 'Eastern Europe', 'PL': 'Eastern Europe', 'RO': 'Eastern Europe', 'RU': 'Eastern Europe', 'SI': 'Eastern Europe', 
    'RS': 'Eastern Europe', 'BA': 'Eastern Europe', 
    'SK': 'Eastern Europe', 'UA': 'Eastern Europe', 'HR': 'Eastern Europe', 'GR': 'Eastern Europe',  'MD': 'Eastern Europe',
    'AE': 'Middle East', 'AM': 'Middle East', 'AZ': 'Middle East', 'BH': 'Middle East', 'GE': 'Middle East', 'IL': 'Middle East', 
    'IQ': 'Middle East', 'IR': 'Middle East', 'JO': 'Middle East', 'KW': 'Middle East', 'LB': 'Middle East', 'OM': 'Middle East', 
    'QA': 'Middle East', 'SA': 'Middle East', 'SY': 'Middle East', 'TR': 'Middle East', 'YE': 'Middle East', 'CY': 'Middle East', 
    'DZ': 'Africa', 'EG': 'Africa', 'KE': 'Africa', 'MA': 'Africa', 'NG': 'Africa', 'ZA': 'Africa', 'TN': 'Africa', 'UG': 'Africa', 
    'GH': 'Africa', 'MU': 'Africa', 'CF': 'Africa',
    'CN': 'Asia', 'IN': 'Asia', 'ID': 'Asia', 'JP': 'Asia', 'KR': 'Asia', 'MY': 'Asia', 'PH': 'Asia', 'SG': 'Asia', 'TH': 'Asia', 
    'VN': 'Asia', 'HK': 'Asia', 'TW': 'Asia', 'PK': 'Asia', 'UZ': 'Asia',
    'AU': 'Oceania', 'NZ': 'Oceania', 'AS': 'Oceania'
}

# This function is for categorizing job titles into categories.
def categorize_job_title(job_title):
    if re.search(r'\b(Data Science|Data Analyst|Analytics|Insight Analyst)\b', job_title, re.IGNORECASE):
        return 'Data Science & Analytics'
    elif re.search(r'\b(Machine Learning|ML|Deep Learning|Computer Vision|NLP|AI)\b', job_title, re.IGNORECASE):
        return 'Machine Learning & AI'
    elif re.search(r'\b(Business Intelligence|BI Developer|BI Analyst)\b', job_title, re.IGNORECASE):
        return 'Business Intelligence'
    elif re.search(r'\b(Data Engineer|Data Architect|ETL|Data Integration)\b', job_title, re.IGNORECASE):
        return 'Data Engineering'
    elif re.search(r'\b(Research Scientist|Research Engineer|Research Analyst)\b', job_title, re.IGNORECASE):
        return 'Research'
    elif re.search(r'\b(Head of Data|Data Manager|Operations|Director|Lead)\b', job_title, re.IGNORECASE):
        return 'Management & Operations'
    elif re.search(r'\b(Software Engineer|Developer|Technician|Software Data Engineer)\b', job_title, re.IGNORECASE):
        return 'Software Engineering'
    else:
        return 'Others'