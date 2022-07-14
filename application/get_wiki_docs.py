# %%
import pandas as pd
import wikipedia

# %% get references from statistics and Machine Learning

mp1 = wikipedia.page("Statistics")
mp2 = wikipedia.page("Machinelearning")


# %% delete non meaningful pages
mp1_links = mp1.links
mp2_links = mp2.links

non_meaninfull_link = [
    "CiteSeerX (identifier)",
    "ArXiv",
    "ArXiv (identifier)",
    "Springer Science+Business Media",
    "Springer Nature",
    "OCLC (identifier)",
    "S2CID (identifier)",
    "Sexual selection",
    "Temperature",
    "Western Electric Company",
    "Open textbook",
    "Oikonyms in Western and South Asia",
    "PMC (identifier)",
    "PMID (identifier)",
    "Longitude",
    "List of academic statistical associations",
    "List of fields of application of statistics",
    "List of important publications in statistics",
    "List of national and international statistical services",
    "List of open source statistical packages",
    "List of statistical packages",
    "List of statisticians",
    "List of statistics articles",
    "List of statistics journals",
    "List of university statistical consulting centers",
    "Lists of mathematics topics",
    "Lists of statistics topics",
    "Process art",
    "Iannis Xenakis",
    "Ibn Adlan",
    "Islamic Golden Age",
    "Professional certification in financial services",
    "ISBN (identifier)",
    "ISSN (identifier)",
    "Glossary of Arabic toponyms",
    "Glossary of aerospace engineering",
    "Glossary of agriculture",
    "Glossary of archaeology",
    "Glossary of architecture",
    "Glossary of areas of mathematics",
    "Glossary of artificial intelligence",
    "Glossary of astronomy",
    "Glossary of biology",
    "Glossary of bird terms",
    "Glossary of botany",
    "Glossary of calculus",
    "Glossary of chemistry terms",
    "Glossary of civil engineering",
    "Glossary of clinical research",
    "Glossary of computer hardware terms",
    "Glossary of computer science",
    "Glossary of ecology",
    "Glossary of economics",
    "Glossary of electrical and electronics engineering",
    "Glossary of engineering: A–L",
    "Glossary of engineering: M–Z",
    "Glossary of entomology terms",
    "Glossary of environmental science",
    "Glossary of evolutionary biology",
    "Glossary of genetics",
    "Glossary of geography terms",
    "Glossary of geology",
    "Glossary of ichthyology",
    "Glossary of machine vision",
    "Glossary of mathematical symbols",
    "Glossary of mechanical engineering",
    "Glossary of medicine",
    "Glossary of meteorology",
    "Glossary of nanotechnology",
    "Glossary of physics",
    "Glossary of probability and statistics",
    "Glossary of psychiatry",
    "Glossary of robotics",
    "Glossary of scientific naming",
    "Glossary of structural engineering",
    "Glossary of virology",
]

mp1_links = [x for x in mp1_links if x not in non_meaninfull_link]
mp2_links = [x for x in mp2_links if x not in non_meaninfull_link]


# %% investigation of mainpages

# len(mp1.links)
# len(mp2.links)
# mp1.summary
# mp2.summary

# %% get summaries of subpages

list_errors = []

list_pageid = []
list_text = []
list_title = []
list_statistics = []
list_ml = []

for i, link in enumerate(mp1_links):
    if i % 100 == 0:
        print(i)
    try:
        sp = wikipedia.page(link)
        list_pageid.append(sp.pageid)
        list_text.append(sp.summary)
        list_title.append(sp.title)
        list_statistics.append(1)
        list_ml.append(0)
    except:
        try:
            link_c = link.replace(" ", "")
            sp = wikipedia.page(link_c)
            list_pageid.append(sp.pageid)
            list_text.append(sp.summary)
            list_title.append(sp.title)
            list_statistics.append(1)
            list_ml.append(0)

        except:
            print(link)
            list_errors.append(link)

for link in mp2_links:
    try:
        sp = wikipedia.page(link)
        list_pageid.append(sp.pageid)
        list_text.append(sp.summary)
        list_title.append(sp.title)
        list_statistics.append(1)
        list_ml.append(0)
    except:
        try:
            link_c = link.replace(" ", "")
            sp = wikipedia.page(link_c)
            list_pageid.append(sp.pageid)
            list_text.append(sp.summary)
            list_title.append(sp.title)
            list_statistics.append(1)
            list_ml.append(0)

        except:
            print(link)
            list_errors.append(link)

# %% substitute non-found
list_errors


# %%

corpus_with_metadata = pd.DataFrame(
    {
        "pageid": list_pageid,
        "text": list_text,
        "title": list_title,
        "statistics": list_statistics,
        "ml": list_ml,
    }
)

# %% check for duplicates

mask = corpus_with_metadata.duplicated(subset=["pageid"], keep=False)
corpus_with_metadata.loc[mask, ["statistics", "ml"]] = 1
corpus_with_metadata.drop_duplicates(inplace=True)


# %% save file

corpus_with_metadata.to_csv("application/data/wiki_corpus.csv")
