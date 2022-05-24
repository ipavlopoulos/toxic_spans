# The Dataset (Toxic Spans)
**TOXICSPANS** contains the 11,035 posts we annotated for toxic spans. The unique posts are actually 11,006, since a few were duplicates and were removed in subsequent experiments.
A few other posts were used as quiz questions to check the reliability of candidate annotators and were also discarded in subsequent experiments.

We used posts (comments) from the publicly available Civil Comments dataset (Borkan et al., 2019), which already provides whole-post toxicity annotations. We followed the toxicity definition that was used in Civil Comments, i.e., we use ‘toxic’
as an umbrella term that covers abusive language phenomena, such as insults, hate speech, identity attack, or profanity. This definition of toxicity has
been used extensively in previous work (Hosseini et al., 2017; Van Aken et al., 2018; Karan and Šnajder, 2019; Han and Tsvetkov, 2020; Pavlopoulos et al., 2020). We asked crowd annotators to
highlight the spans that constitute “anything that is rude, disrespectful, or unreasonable that would make someone want to leave a conversation”. Besides toxicity our annotators were also asked to
select a subtype for each highlighted span, choosing between insult, threat, identity-based attack, profane/obscene, or other toxicity. Asking the annotators to also select a category was intended as a
priming exercise to increase their engagement, but it may have also helped them align their notions of toxicity further, increasing inter-annotator agreement. For the purposes of our experiments, we
collapsed all the subtypes into a single toxic class, and we did not study them further; but the subtypes are included in the new dataset we release.

### Annotation
From the original Civil Comments dataset (1.2M posts), we retained only posts that had been found toxic by at least half of the crowdraters. This left approx. 30k toxic posts. We
selected a random 11k subset of the 30k posts for toxic spans annotation. We used the crowdannotation platform of Appen.2 We employed three crowd-raters per post, all of whom were warned
for explicit content. Raters were selected from the smallest group of the most experienced and accurate contributors. The raters were asked to mark the toxic word sequences (spans) of each post by
highlighting each toxic span on their screen. For each post, the dataset includes the spans of all three raters. If the raters believed a post was not actually toxic, or that the entire post would have to be annotated, they were instructed to select appropriate
tick-boxes in the interface, without highlighting any span. The tick-boxes were separate and the dataset shows when (if) any of the two were ticked. Hence, when no toxic spans are provided (for a particular post by a particular rater), it is clear if the
rater thought that the post was not actually toxic, or that the entire post would have to be annotated.

It is not possible to annotate toxic spans for every toxic post. For example, in some posts the core message being conveyed may be inherently toxic (e.g., a sarcastic post indirectly claiming that people
of a particular origin are inferior) and, hence, it may be difficult to attribute the toxicity of those posts to particular spans. In such cases, the posts may end up having no toxic span annotations, according
to the guidelines given to the annotators. In other cases, however, it is easier to identify particular spans (possibly multiple per post) that make a post toxic,
and these toxic spans often cover only a small part of the post.

### Ground truth
To obtain the ground truth of our dataset, we averaged the labels per character of the annotators per post. We used the following process:
for each post *t*, first we mapped each annotated span of each rater to its character offsets. We then assigned a toxicity score to each character offset of *t*, computed as the fraction of raters who annotated
that character offset as toxic (included it in their toxic spans). We retained only character offsets with toxicity scores higher than 50%; i.e., at least two raters must have included each character offset
in their spans

The dataset is stored as a CSV. The data file contains 7 columns:
* probability = a dict with the first and the last character offsets of each token (that was rated by at least one annotator as toxic) as a key, and  the average toxicity as a value
* position = the character offsets of all the toxic spans(avg toxicity > 50%) found by the annotators (ground truth)
* text = the average toxicity of each token that was rated by at least one annotator as toxic 
* type = the type of toxicity of each toxic span 
* support = the number of annotators per post 
* text_of_post = the text of the post 
* position_probability = the average toxicity of each character offset that was found by at least one annotator as toxic  
