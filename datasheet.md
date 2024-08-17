# Datasheet for dataset "Better Inductive KGC Datasets"

Questions from the [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) paper, v7.

Markdown template from [this repo](https://github.com/fau-masters-collected-works-cgarbin/datasheet-for-dataset-template). 

Jump to section:

- [Motivation](#motivation)
- [Composition](#composition)
- [Collection process](#collection-process)
- [Preprocessing/cleaning/labeling](#preprocessingcleaninglabeling)
- [Uses](#uses)
- [Distribution](#distribution)
- [Maintenance](#maintenance)

## Motivation

<!-- _The questions in this section are primarily intended to encourage dataset creators
to clearly articulate their reasons for creating the dataset and to promote transparency
about funding interests._ -->

### For what purpose was the dataset created? 

<!-- _Was there a specific task in mind? Was there a specific gap that needed to be filled?
Please provide a description._ -->

This dataset was created to facilitate better benchmarking of inductive knowledge graph (KG) completion. The task is to train a method on one KG and then test on a separate KG. This KG is typically contains disjoint entities. They also may contain the same set of relations or a number of unseen relations. While existing datasets do exist for this task, we find that we can achieve strong performance on them using the Personalized PageRank (PPR) score. This allows for a simple shortcut to exist, wheras models can bypass the relational information predict new facts. In this dataset, we aimed to mitigate this shortcut through the introduction of a newer method of constructing inductive datasets.

### Who created the dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)?

The dataset was created by Harry Shomer (Michigan State University), Jay Revolinsky (Michigan State University), and Jiliang Tang (Michigan State University).

### Who funded the creation of the dataset? 

The authors were funded by several agencies including the National Science Foundation (NSF),
the Army Research Office (ARO), the Home Depot, Cisco Systems Inc, Amazon Faculty Award,
Johnson&Johnson, JP Morgan Faculty Award and SNAP.

### Any other comments?

None.

## Composition
<!-- 
_Most of these questions are intended to provide dataset consumers with the
information they need to make informed decisions about using the dataset for
specific tasks. The answers to some of these questions reveal information
about compliance with the EU’s General Data Protection Regulation (GDPR) or
comparable regulations in other jurisdictions._ -->

### What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)?
<!-- 
_Are there multiple types of instances (e.g., movies, users, and ratings; people and
interactions between them; nodes and edges)? Please provide a description._ -->

Each dataset contain a training graph, $k$ testing graphs, a set of validation samples, and a set of test samples. Each sample is of the form $(s, r, o)$ where $s$ and $o$ are entities (i.e., nodes) in the graph and $r$ is a relation (i.e., edge type).

### How many instances are there in total (of each type, if appropriate)?

We split the statistics up by training and testing graphs in the table below:

| Dataset | Graph | \# Edges | \# Entities | \# Rels | \# Valid/Test |
|---------------|-------------|-----------------|-------------------|---------------|---------------------|
| CoDEx-M (E)              | Train       | 76,960          | 8,362             | 47            | 8,552               |
| CoDEx-M (E)               | Test 1 | 69,073          | 8,003             | 40            | 7,674               |
| WN18RR (E)              | Train       | 24,584          | 12,142            | 11            | 2,458               |
| WN18RR (E)              | Test 1 | 18,258          | 8,660             | 10            | 1,831               |
| WN18RR (E)              | Test 2 | 5,838           | 2975              | 8             | 572                 |
| HetioNet (E)             | Train       | 101,667         | 3,971             | 14            | 11,271              |
| HetioNet (E)              | Test 1 | 49,590          | 2,279             | 11            | 5,490               |
| HetioNet (E)              | Test 2 | 37,927          | 2455              | 12            | 4,187               |
|  FB15k-237 (E, R)             | Train       | 45,597          | 2,869             | 105           | 5,062               | 
| FB15k-237 (E, R)               | Test 1 | 35,937          | 1,835             | 143           | 3,992               | 
|  FB15k-237 (E, R)              | Test 2 | 51,693          | 2,606             | 143           | 5,735               | 
|  CoDEx-M (E, R)             | Train       | 29,634          | 4,038             | 36            | 3293                | 
|  CoDEx-M (E, R)             | Test 1 | 70,137          | 7,938             | 39            | 7,794               | 
|  CoDEx-M (E, R)             | Test 2 | 8,821           | 2,606             | 28            | 979                 |           



### Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set?

<!-- _If the dataset is a sample, then what is the larger set? Is the sample representative
of the larger set (e.g., geographic coverage)? If so, please describe how this
representativeness was validated/verified. If it is not representative of the larger set,
please describe why not (e.g., to cover a more diverse range of instances, because
instances were withheld or unavailable)._ -->

These datasets are sampled from existing transductive datasets. As such, they inheritely only contain a subset of the potential samples. However, we note that perfectly splitting the transductive dataset into inductive graphs without the loss of information would be extremely rare due to the nature of how these graphs are structured.

### What data does each instance consist of? 

<!-- _“Raw” data (e.g., unprocessed text or images) or features? In either case, please
provide a description._ -->

Each instance is a single edge (also known as a "triple" or "fact"). It contains two entities and one relation.

### Is there a label or target associated with each instance?

No. We only provide positive samples. During training negative samples are randomly sampled. During evaluation, we sample all possible negative corruptions of the given positive sample. 

### Is any information missing from individual instances?

<!-- _If so, please provide a description, explaining why this information is missing (e.g.,
because it was unavailable). This does not include intentionally removed information,
but might include, e.g., redacted text._ -->

No.

### Are relationships between individual instances made explicit (e.g., users’ movie ratings, social network links)?

<!-- _If so, please describe how these relationships are made explicit._
 -->

Yes. For example, instances that share an entity will be linked in the graph.

### Are there recommended data splits (e.g., training, development/validation, testing)?

<!-- _If so, please provide a description of these splits, explaining the rationale behind them._ -->

Yes. They were chosen randomly from either the train or testing graphs.

### Are there any errors, sources of noise, or redundancies in the dataset?

<!-- _If so, please provide a description._ -->

It is possible, however they would be from the original transductive dataset and not introduced by our process.

### Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)?

<!-- _If it links to or relies on external resources, a) are there guarantees that they will
exist, and remain constant, over time; b) are there official archival versions of the
complete dataset (i.e., including the external resources as they existed at the time the
dataset was created); c) are there any restrictions (e.g., licenses, fees) associated with
any of the external resources that might apply to a future user? Please provide descriptions
of all external resources and any restrictions associated with them, as well as links or other
access points, as appropriate._ -->

Yes, it is self-contained.

### Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor-patient confidentiality, data that includes the content of individuals’ non-public communications)?

<!-- _If so, please provide a description._
 -->

No.

### Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?

<!-- _If so, please describe why._ -->

No.

### Does the dataset relate to people? 

<!-- _If not, you may skip the remaining questions in this section._ -->

No.

### Does the dataset identify any subpopulations (e.g., by age, gender)?

<!-- _If so, please describe how these subpopulations are identified and provide a description of
their respective distributions within the dataset._ -->

N/A

### Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other data) from the dataset?

<!-- _If so, please describe how._ -->

N/A

### Does the dataset contain data that might be considered sensitive in any way (e.g., data that reveals racial or ethnic origins, sexual orientations, religious beliefs, political opinions or union memberships, or locations; financial or health data; biometric or genetic data; forms of government identification, such as social security numbers; criminal history)?

<!-- _If so, please provide a description._ -->

N/A

### Any other comments?

None.

## Collection process

<!-- _\[T\]he answers to questions here may provide information that allow others to
reconstruct the dataset without access to it._ -->

### How was the data associated with each instance acquired?
<!-- 
_Was the data directly observable (e.g., raw text, movie ratings), reported by subjects (e.g.,
survey responses), or indirectly inferred/derived from other data (e.g., part-of-speech tags,
model-based guesses for age or language)? If data was reported by subjects or indirectly
inferred/derived from other data, was the data validated/verified? If so, please describe how._ -->

The data was extracted from existing transductive KGs. This includes: 
1. FB15k-237
2. WN18RR
3. CoDEx-M
4. HetioNet

See our paper for more details on each and how to access the original datasets.

### What mechanisms or procedures were used to collect the data (e.g., hardware apparatus or sensor, manual human curation, software program, software API)?

<!-- _How were these mechanisms or procedures validated?_ -->

Since we did not construct the original transductive datasets, we only had to download the data from their sources.

### If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)?

The graph partitioning scheme and the determination of which samples are validation/test are done randomly. The strategy is done with a fixed seed, ensuring that others can generate the same dataset on their machine.

### Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)?

The splits were constructed by the authors of this work.

### Over what timeframe was the data collected?

<!-- _Does this timeframe match the creation timeframe of the data associated with the instances (e.g.
recent crawl of old news articles)? If not, please describe the timeframe in which the data
associated with the instances was created._ -->

Please see the how the original transductive datasets were created.

### Were any ethical review processes conducted (e.g., by an institutional review board)?

<!-- _If so, please provide a description of these review processes, including the outcomes, as well as
a link or other access point to any supporting documentation._ -->

No.

### Does the dataset relate to people?

<!-- _If not, you may skip the remainder of the questions in this section._ -->

No.

### Did you collect the data from the individuals in question directly, or obtain it via third parties or other sources (e.g., websites)?

N/A

### Were the individuals in question notified about the data collection?

<!-- _If so, please describe (or show with screenshots or other information) how notice was provided,
and provide a link or other access point to, or otherwise reproduce, the exact language of the
notification itself._ -->

N/A

### Did the individuals in question consent to the collection and use of their data?

<!-- _If so, please describe (or show with screenshots or other information) how consent was
requested and provided, and provide a link or other access point to, or otherwise reproduce, the
exact language to which the individuals consented._ -->

N/A

### If consent was obtained, were the consenting individuals provided with a mechanism to revoke their consent in the future or for certain uses?
<!-- 
_If so, please provide a description, as well as a link or other access point to the mechanism
(if appropriate)._ -->

N/A

### Has an analysis of the potential impact of the dataset and its use on data subjects (e.g., a data protection impact analysis) been conducted?
<!-- 
_If so, please provide a description of this analysis, including the outcomes, as well as a link
or other access point to any supporting documentation._ -->

N/A

### Any other comments?

None.

## Preprocessing/cleaning/labeling

<!-- _The questions in this section are intended to provide dataset consumers with the information
they need to determine whether the “raw” data has been processed in ways that are compatible
with their chosen tasks. For example, text that has been converted into a “bag-of-words” is
not suitable for tasks involving word order._ -->

### Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)?

<!-- _If so, please provide a description. If not, you may skip the remainder of the questions in
this section._ -->

None.

### Was the “raw” data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)?

<!-- _If so, please provide a link or other access point to the “raw” data._ -->

Yes, our code automatically fetches and saves the original raw data.

### Is the software used to preprocess/clean/label the instances available?

<!-- _If so, please provide a link or other access point._ -->

Yes. See [here](https://github.com/HarryShomer/Better-Inductive-KGC).

### Any other comments?

None.

## Uses

<!-- _These questions are intended to encourage dataset creators to reflect on the tasks
for which the dataset should and should not be used. By explicitly highlighting these tasks,
dataset creators can help dataset consumers to make informed decisions, thereby avoiding
potential risks or harms._ -->

### Has the dataset been used for any tasks already?

<!-- _If so, please provide a description._ -->

Yes, the datasets are solely meant for the use of inductive knowledge graph completion and have been used for such.

### Is there a repository that links to any or all papers or systems that use the dataset?

<!-- _If so, please provide a link or other access point._ -->

None.

### What (other) tasks could the dataset be used for?

We are unsure, as datasets for KG completion are typically only used for that task. However, it could be that our dataset finds use in other KG-related tasks like entity linking.

### Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?

<!-- _For example, is there anything that a future user might need to know to avoid uses that
could result in unfair treatment of individuals or groups (e.g., stereotyping, quality of
service issues) or other undesirable harms (e.g., financial harms, legal risks) If so, please
provide a description. Is there anything a future user could do to mitigate these undesirable
harms?_ -->

None.

### Are there tasks for which the dataset should not be used?

<!-- _If so, please provide a description._ -->

Nothing specific.

### Any other comments?

None.

## Distribution

### Will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created? 

<!-- _If so, please provide a description._ -->

No.

### How will the dataset will be distributed (e.g., tarball on website, API, GitHub)?

<!-- _Does the dataset have a digital object identifier (DOI)?_ -->

It is available on Github. See the repo [here](https://github.com/HarryShomer/Better-Inductive-KGC).

### When will the dataset be distributed?    

It is currently available.

### Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?

<!-- _If so, please describe this license and/or ToU, and provide a link or other access point to,
or otherwise reproduce, any relevant licensing terms or ToU, as well as any fees associated
with these restrictions._ -->

It is currently made available under the MIT License. As such, there are no restrictions placed on the use of the dataset.

### Have any third parties imposed IP-based or other restrictions on the data associated with the instances?

<!-- _If so, please describe these restrictions, and provide a link or other access point to, or
otherwise reproduce, any relevant licensing terms, as well as any fees associated with these
restrictions._ -->

No.

### Do any export controls or other regulatory restrictions apply to the dataset or to individual instances?
<!-- 
_If so, please describe these restrictions, and provide a link or other access point to, or otherwise
reproduce, any supporting documentation._ -->

No.

### Any other comments?

None.

## Maintenance

<!-- _These questions are intended to encourage dataset creators to plan for dataset maintenance
and communicate this plan with dataset consumers._ -->

### Who is supporting/hosting/maintaining the dataset?

The dataset is supported by the authors, chiefly -- Harry Shomer.

### How can the owner/curator/manager of the dataset be contacted (e.g., email address)?

shomerha@msu.edu

### Is there an erratum?

<!-- _If so, please provide a link or other access point._ -->

No.

### Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances)?

<!-- _If so, please describe how often, by whom, and how updates will be communicated to users (e.g., mailing list, GitHub)?_ -->

If needed, the authors will update the dataset. These updates will be communicated via Github and displayed in a chaneglog in the README.

### If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances (e.g., were individuals in question told that their data would be retained for a fixed period of time and then deleted)?

<!-- _If so, please describe these limits and explain how they will be enforced._ -->

N/A

### Will older versions of the dataset continue to be supported/hosted/maintained?

<!-- _If so, please describe how. If not, please describe how its obsolescence will be communicated to users._ -->

They will still be available via Github, by looking at earlier versions of the repo.

### If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?

<!-- _If so, please provide a description. Will these contributions be validated/verified? If so,
please describe how. If not, why not? Is there a process for communicating/distributing these
contributions to other users? If so, please provide a description._ -->

Yes. Others are encouraged to submit an issue or a pull request on the [repo](https://github.com/HarryShomer/Better-Inductive-KGC).

### Any other comments?

None.