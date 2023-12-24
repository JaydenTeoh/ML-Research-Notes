# Introduction to Probability

## What is Probability

Probability measures the likelihood of an event occurring, such as the probability of a fair coin landing on heads (50 percent or 1/2 chance) or a dice landing on the number 4 (1/6 chance). 

![Untitled](images/Untitled.png)

To illustrate, consider a school with 10 kids, where 3 play soccer. The probability of randomly picking a soccer player is found by dividing the number of soccer-playing kids by the total children. This is denoted as

$$
P(\text{soccer}) = \frac{\text{no. of kids playing soccer}}{\text{total no. of kids}} = \frac{3}{10} = 0.3
$$

![Untitled](images/Untitled%201.png)

In Venn diagrams, the **sample space (denoted by $S$)** refers to the set of all possible outcomes and the **event (denoted by $E$**) is a subset of $S$ to which a probability is assigned. Therefore, the probability of an event occurring is the following formula

$$
P(E) = \frac{|E|}{|S|}
$$

where $|E|$ and $|S|$ refers to the cardinality (the number of elements in the set) of the event and sample space respectively

Let’s illustrate using another example. Given that we flip a coin 3 times consecutively, what would be the probability that we land 3 heads? One way to calculate this would be to find out all possible outcomes (sample space) and find out the occurrence of landing 3 heads (event).

![Untitled](images/Untitled%202.png)

We get $P(HHH) = \frac{1}{8} = 0.125$

### Complement Rule

Now that we understood how to calculate the probability of an event, how do calculate the probability of the event *not* occurring, also know as the probability of the **complementary event**. Given an event $A$, its complement is denoted by $E'$. In a random experiment, the probabilities of all possible events (the sample space) must total to 1 -  that is, some outcome must occur on every trial. For two events to be complements, they must be collectively exhaustive; together filling the entire sample space. Therefore,

$$
P(E') = 1 - P(E)
$$

This is known as the **complement rule**.

Using the previous example, the probability of not landing 3 heads consecutively is given by

$$
P((HHH)') = 1 - \frac{1}{8} = \frac{7}{8}
$$

### Sum of Probabilities (Disjoint Events)

**Disjoint events** are those that cannot occur simultaneously; if one event happens, the other cannot. Let’s understand this using an example. Let’s say at a school of 10 kids, kids can only play one sport. Therefore we can say the kids that play soccer and the kids that play basketball are events that are **disjoint** or **mutually exclusive**, because there cannot exist a kid that plays both basketball and soccer. 

![Untitled](images/Untitled%203.png)

Because both events are mutually exclusive, if we want to calculate the probability of either of them occurring, we can simply sum up their individual probabilities. As such,

$$
P(\text{soccer or basketball}) = P(\text{soccer}) + P(\text{basketball}) = 0.3 + 0.4 = 0.7
$$

![Untitled](images/Untitled%204.png)

Viewing it using a Venn diagram, we can see that the set of kids that play soccer, $A$, and set of kids that play basketball, $B$, do not intersect at all, which means

$$
\begin{align} \notag
|A\cup B| &= |A| + |B| \\ \notag
\implies P(A\cup B) &= P(A) + P(B)
\end{align}
$$

where $A \cup B$ refers to the union between both events.

### Sum of Probabilities (Joint Events)

Now that we understand how to calculate sum of probabilities for events that are disjoint, how do we do so for events that are joint; events that overlap. Real-world events often overlap, and considering these overlapping outcomes is vital in probability calculations.

Let’s say in the school of 10 kids, kids are allowed to play more than one sport now. 6 kids play soccer and 5 kids play basketball and 3 kids play both. How do we now calculate the probability of kids that play either or both sports? We can simply sum their probabilities up because $0.6 + 0.5 = 1.1 > 1$ → there is a contradiction (the probability of all outcomes must be equal to 1)

Let’s understand it using the Venn diagram.

![Untitled](images/Untitled%205.png)

We can see that the intersection between the event space where a kid plays soccer, $A$, and the event space where a kid plays basketball, $B$. The intersection between these two event spaces is given by $S \cap B$.

We know that 

$$
|A| = 6, |B| = 5, |A \cap B| = 3
$$

To calculate the size of the union of both event spaces, we need to deduct the size of the overlap from the sum of the size of both event spaces

$$
|A \cup B| = |A| + |B| - |A \cap B| = 6 + 5 - 3 = 8
$$

As such, to calculate the probabilities of the union of the two joint events, we have

$$
P(A \cup B) = P(A) + P(B) - P(A \cap B) = 0.5 + 0.6 - 0.3 = 0.8
$$

## Independence

Independence occurs when the occurrence of one event doesn't influence the probability of another event. Understanding independence is crucial in probability and machine learning as assuming independence simplifies calculations and predictions. 

![Untitled](images/Untitled%206.png)

For example, in a fair coin toss, a toss's outcome doesn't affect the following toss. As such, how do we calculate the probability that we land 5 heads consecutively? We could list out all possible outcomes to find the probability of landing 5 consecutive heads but that would be extremely tedious. We know the probability of landing a head is 0.5 and the event of landing a single head is independent from the next, or other words if we land a head, the probability of landing the next head is still 0.5 (unaffected).

As such we can simply multiply their individual probabilities which means

$$
P(HHHHH) = \frac{1}{2} \times \frac{1}{2} \times \frac{1}{2} \times \frac{1}{2} \times \frac{1}{2} = \frac{1}{32}
$$

In general, if two events $A$ and $B$ are independent (often written as $A \perp B$), their joint probability is given by their product of their probabilities, i.e.

$$
P(A \cap B) = P(A) \times P(B)
$$

### Birthday Problem

The birthday problem is a fascinating probability concept that examines the likelihood of shared birthdays among a group of individuals. Initially, one might assume it's unlikely for two individuals in a small group to share a birthday. However, as the group size increases, the probability of shared birthdays surprisingly becomes more probable.

Suppose you have 30 friends at a party (excluding yourself). The question arises: is it more likely that two people share a birthday, or that no two individuals share a birthday within this group? Contrary to intuition, the probability of two people having the same birthday among the 30 friends is approximately 70%, while the probability of no shared birthdays is only around 0.3%. 

![Untitled](images/Untitled%207.png)

To understand this paradoxical outcome, imagine a scenario where each person's birthday is represented by a box in a year (365 days). When there's only one person, the probability of not sharing a birthday is $\frac{365}{365}$. When a new individual joins, the probability that both of them do not share the same birthday is given by $\frac{365}{365} \cdot \frac{364}{365}$ (Note we have to remove 1 birthday from the possible birthdays to choose from). In a group of $n$ individuals, the probability becomes

$$
P(A)={\frac {365}{365}}\times {\frac {364}{365}}\times {\frac {363}{365}}\times {\frac {362}{365}}\times \cdots \times {\frac {365 - n}{365}}
$$

![Untitled](images/Untitled%208.png)

The graph depicting the probability of no shared birthdays against the number of people in the group highlights the rapid decline in the likelihood of no shared birthdays as the group size increases. This probability drops below 0.5 precisely at 23 people, signifying that it's more probable than not for two individuals in a group of this size to share a birthday.

## Conditional Probability

In probability theory, **conditional probability** is a measure of the probability of an event occurring, given that another event (by assumption, presumption, assertion or evidence) is **already known to have occurred**. 

Let’s understand using an example. Given that the first coin toss is heads, what is the probability that we land on heads twice?

![Untitled](images/Untitled%209.png)

The original probability, when both coins were tossed independently, was $\frac{1}{4}$. However, given that the first coin landed as heads, the sample space changes, making the probability of getting two heads among the remaining possibilities as $\frac{1}{2}$. This shift in probability is termed conditional probability and it is denoted by

$$
P(HH|\text{1st is H}) = \frac{1}{2}
$$

In general, given two events $A$ and $B$, if the event of interest is $B$ and the event $A$ is known to have occurred, the conditional probability of $ B$ given $ A$ is written as $P(B|A)$.

This can also be understood as the fraction of probability $A$ that intersects with $B$

$$
P(B \cap A) = P(B|A) \times P(A)
$$

![Untitled](images/Untitled%2010.png)

Imagine you have a school with 100 kids. 40% of them play soccer and 80% of the kids who play soccer wears running shoes. What is the probability that a kid wear running shoes, $R$, and plays soccer, $S$?

$$
P(R \cap S) = P(R |S) \times P(S) = 0.8 \times 0.4 = 0.32
$$

If we calculate the number of kids who play soccer and wear running shoes, we indeed get 32 out of 100 kids.

If $B$ is independent from $A$, for example in the coin toss we see earlier, we get

$$
P(B|A) = P(B)
$$

## Bayes’ Theorem

Bayes' theorem is a fundamental principle in probability theory used to calculate conditional probabilities by considering prior knowledge or information.

### Intuition

Let's consider a scenario to illustrate Bayes' theorem: Imagine there's a rare disease affecting 1 in every 10,000 people in a population of 1 million. You undergo a test for this disease, and the test is reported as positive by your doctor, which might be alarming. However, before jumping to conclusions, let's apply Bayes' theorem to determine the probability of actually having the disease given a positive test result.

Here are the critical numbers:

- 1 million people in the population.
- 1 in 10,000 are affected by the disease → 100 sick individuals.
- The test is 99% effective (correctly identifies 99 out of 100 sick individuals and misdiagnoses 1 out of 100 healthy individuals).

![Untitled](images/Untitled%2011.png)

When all 1 million individuals are considered, only 100 are sick, and 999,900 are healthy. The test result classifies people into four groups:

- Sick and correctly diagnosed as sick: 99 individuals (99% of sick people).
- Sick but misdiagnosed as healthy: 1 individual (1% of sick people).
- Healthy but misdiagnosed as sick: 9,999 individuals (1% of healthy people).
- Healthy and correctly diagnosed as healthy: 989,901 individuals (99% of healthy people).

Now, let's focus on the group that matters: those who tested positive (99 + 9,999 individuals). Among this group, the probability of actually being sick is calculated as 

$$
P(\text{sick} | \text{diagnosed sick}) = \frac{\text{sick and diagnosed sick} = 99}{\text{diagnosed sick} = 9999 + 99} = \frac{99}{10098} = 0.0098
$$

Surprisingly, the probability that you are actually sick is found to be less than 1%.

This example demonstrates the essence of Bayes' theorem, showcasing how prior probabilities, along with new evidence or test outcomes, can help refine and revise the probability of an event.

### Formula

The formula for Bayes' theorem is as follows:

$$
P(A|B) = \frac{P(A \cap B)}{P(B)} = \frac{P(A) \times P(B|A)}{P(B)}
$$

Where:

- $P(A|B)$ is the probability of event $A$ occurring given that event $B$ has occurred.
- $P(A)$ is the prior probability of event $A$ occurring without considering $B$.
- $P(B|A)$ is the conditional probability of event $B$ occurring given that event $A$ has occurred.
- $P(B)$ is the prior probability of event $B$ occurring without considering $A$.

Let’s take a look again at the example of a rare disease. Given

- $P(\text{sick}) = 0.0001$: 1 in 10,000 get sick
- $P(\text{not sick}) = 0.9999$: 9999 in 10,000 are not sick
- $P(\text{diagnosed sick}|\text{sick}) = 0.99$: diagnosis correctly identifies sick person 99% of the time
- $P(\text{diagnosed sick}|\text{not sick}) = 0.01$: misdiagnosis happens 1% of the time

We want to find out

$$
\begin{align} \notag
P(\text{sick} | \text{diagnosed sick}) 
&= \frac{P(\text{sick} \cap \text{diagnosed sick})}{P(\text{diagnosed sick})}\\\ \notag
&= \frac{P(\text{sick}) \times P(\text{diagnosed sick}|\text{sick})}{P(\text{diagnosed sick})}\\\ \notag
&= \frac{P(\text{sick}) \times P(\text{diagnosed sick}|\text{sick})}{P(\text{sick}\cap \text{diagnosed sick}) + P(\text{not sick}\cap \text{diagnosed sick})}\\\ \notag
&= \frac{P(\text{sick}) \times P(\text{diagnosed sick}|\text{sick})}{P(\text{sick}) \times P(\text{diagnosed sick}|\text{sick}) + P(\text{not sick}) \times P(\text{diagnosed sick}|\text{not sick})}\\\ \notag
&= \frac{0.0001 \times 0.99}{(0.0001 \times 0.99) + (0.9999 \times 0.01)}\\\ \notag
&= 0.0098
\end{align}
$$

We can see that the Bayes’ formula allows to derive the correct conditional probability using only the probabilities. We also see that

$$
P(A|B) = \frac{P(A \cap B)}{P(B)} = \frac{P(A \cap B)}{P(A \cap B) + P(A' \cap B)}
$$

### Prior and Posterior

- **Prior Probability $P(A)$**: This refers to the initial probability of an event occurring based on the available information or assumptions, without considering any specific new data or evidence. It represents the original estimation or baseline probability of an event.
- **Posterior Probability $P(A|E)$**: An event $E$ occurs, providing additional information or evidence, leading to the recalculation of the probability. The posterior probability is the updated probability of an event occurring, taking into account the new information from the event.

### Naive Bayes Classifiers

Naive Bayes is a simple technique for constructing classifiers. The core idea behind Naive Bayes is to use Bayes' theorem to calculate the probability of a certain class or label given the presence of particular features. It assumes that features are **conditionally independent** given the class label, even though this might not hold true in real-world scenarios. This simplifying assumption allows for easier computation and often works well in practice, especially in text classification problems.

Given data with features $x_1, x_2, \dots,x_n$, we want to calculate the probability that it belongs to the class $C_k$ (one of the $K$ classes). Using Bayes Theorem, the conditional probability can be written as:

$$
P(C_k|x_1,\dots,x_n) = \frac{P(x_1,x_2, \dots,x_n|C_k) \times P(C_k)}{P(x_1,x_2,\dots,x_n)}
$$

The Naive Bayes algorithm makes the primary assumption that all features are conditionally independent of each other given the class label. In other words, the presence of a particular feature in a class is independent of the presence of any other feature, given the class label. As such, we can express the joint probability of observing all features given class $C_k$ as the product of individuals feature probabilities given $C_k$ which means

$$
P(x_1,x_2, \dots,x_n|C_k) = P(x_1 |C_k) \times P(x_2|C_k) \times \dots \times P(x_n|C_k)
$$

Moreover, we know that the probability of observing features $x_1, x_2, \dots ,x_n$ can be represented as

$$
\begin{align} \notag
P(x_1,x_2,\dots,x_n) &= P(x_1,x_2,\dots,x_n|C_k) \times P(C_k) + P(x_1,x_2,\dots,x_n|C_k') \times (P(C_k')\\\ \notag
& = \sum_{i=1}^m P(x_1,x_2,\dots,x_n | C_i) \times P(C_i)\\\ \notag
& = \sum_{i=1}^m P(x_1| C_i) \times P(x_2| C_i) \times \dots \times P(x_n| C_i)  \times P(C_i)
\end{align}
$$

As such, using Naive Bayes assumption, the original conditional probability can be expressed as following

$$
P(C_k|x_1,\dots,x_n) = \frac{P(x_1 |C_k) \times P(x_2|C_k) \times \dots \times P(x_n|C_k) \times P(C_k)}{\sum_{i=1}^{m} P(x_1| C_i) \times P(x_2| C_i) \times \dots \times P(x_n| C_i)  \times P(C_i)}
$$

This seems complicated but let’s understand using an example. Suppose we have an email dataset containing spam and non-spam (ham) emails. We want to build a Naive Bayes classifier to predict the probability an email is spam based on the occurrence of two words: "lottery" and "winning.” → $P(\text{spam}| \text{lottery, winning})$

![Untitled](images/Untitled%2012.png)

Training Data:

- Total Emails: 100 (20 are spam, 80 are ham)
- Occurrence of the word "lottery":
    - 14 out of 20 spam emails contain "lottery" → $P(\text{lottery}|\text{spam}) = 0.7$
    - 10 out of 80 ham emails contain "lottery" → $P(\text{lottery}|\text{ham}) = 0.125$
- Occurrence of the word "winning":
    - 15 out of 20 spam emails contain "winning" → $P(\text{winning}|\text{spam}) = 0.75$
    - 8 out of 80 ham emails contain "winning" → $P(\text{winning}|\text{ham}) = 0.1$

We want to calculate the following conditional probability

$$
\begin{align} \notag
P(\text{spam}| \text{lottery, winning}) &= \frac{P(\text{lottery, winning} | \text{spam}) \times P(\text{spam})}{P(\text{lottery, winning})}\\\ \notag
&= \frac{P(\text{lottery, winning} | \text{spam}) \times P(\text{spam})}{P(\text{lottery, winning} | \text{spam}) \times P(\text{spam}) + P(\text{lottery, winning} | \text{ham}) \times P(\text{ham})}
\end{align}
$$

We aim to predict whether an email containing both “lottery” and “winning” is spam. Obviously, the features “lottery” and “winning” are not independent as we can see that many emails have both words in them. However, using our Naive Bayes assumption that they are conditionally independent, we can calculate their joint probability using product of their individual probabilities. As such, we can expand the above equation using  Naive Bayes and derive

$$
\begin{align} \notag
P(\text{spam}| \text{lottery, winning}) &= \frac{P(\text{lottery} | \text{spam}) \times P(\text{winning} | \text{spam}) \times P(\text{spam})}{P(\text{lottery} | \text{spam}) \times P(\text{winning} | \text{spam}) \times P(\text{spam}) + P(\text{lottery} | \text{ham}) \times P(\text{winning} | \text{ham}) \times P(\text{ham})}\\\ \notag
&= \frac{0.2 \times 0.7 \times 0.75}{(0.2 \times 0.7 \times 0.75) + (0.8 \times 0.125 \times 0.1)}\\\ \notag
&= 0.913
\end{align}
$$

## Random Variables

A **random variable**, usually written $ X$, is a variable whose possible values are numerical outcomes of a random phenomenon. 

The term 'random variable' can be misleading as its mathematical definition is not actually random nor a variable, but rather it is a function from possible outcomes (e.g., the possible upper sides of a flipped coin such as heads $H$ and tails $T$) in a sample space (e.g., the set $\set{H,T}$) to a measurable space (e.g., $\set{0,1}$) in which 1 is corresponding to $H$ and 0 is corresponding to $T$, respectively), often to the real numbers.

There are two types of random variables, **discrete** and **continuous**.

### Discrete Random Variables

![Untitled](images/Untitled%2013.png)

A **discrete random variable** is one which may take on only a **countable** number of distinct values such as 0,1,2,3,4,........

### Continuous Random Variables

Formally, a continuous random variable is a random variable whose cumulative distribution function is continuous everywhere. There are no "gaps", which would correspond to numbers which have a finite probability of occurring.

## Probability Distributions

### Discrete Probability Distributions

The **probability distribution** of a discrete random variable is a list of probabilities associated with each of its possible values. A **probability mass function** (PMF) is a function that gives the probability that a discrete random variable is exactly equal to some value. The probabilities in the probability distribution of a random variable $X$ must satisfy the following two conditions:

- Each probability $P(x)$ must be between 0 and 1:
    
$$
0 \leq P(x) \leq 1
$$
    
- The sum of all the possible probabilities is :
    
$$
\sum P(x) = 1
$$
    

### Binomial Distributions

![Untitled](images/Untitled%2014.png)

The binomial distribution, a fundamental concept in probability, is best illustrated through coin tosses. Consider flipping a coin 10 times—how many heads could emerge? You could have 0, 1, 2, 3, all the way to 10 heads, each with its corresponding probability, shaping the binomial distribution's histogram. This distribution, a type of discrete distribution, showcases probabilities for discrete outcomes, in this case, the number of heads from multiple coin flips.

![Untitled](images/Untitled%2015.png)

![Untitled](images/Untitled%2016.png)

Graphing the PMF for varying probabilities illustrates its symmetry—symmetrical distributions result from equal probabilities for heads and tails ($p=\frac{1}{2}$). Conversely, biased coins ($p \ne \frac{1}{2}$) create skewed distributions, favouring certain outcomes.

In general, if the random variable $X$ follows the binomial distribution with parameters $n \in \N$ and $p \in [0, 1]$, we write $X \sim B(n,p)$. The probability of getting $k$ successes in $n$ independent Bernoulli trials (with the same rate $p$) is given by the PMF:

$$
P(k:n,p)={\binom {n}{k}}p^{k}(1-p)^{n-k}
$$

The binomial coefficient ${\binom {n}{k}}$ is the number of ways of picking $k$ **unordered** successes from $n$ trials (e.g. number of possible combinations of getting 4 $H$ from 10 coin flips). It is often denoted as $^{n}C_k$ or “n choose k”. This coefficient can be computed using the following formula

$$
{\binom {n}{k}} = \frac{n !}{k!(n-k)!}
$$

### Bernoulli Distribution

The Bernoulli distribution is the discrete probability distribution of a random variable having two possible outcomes labelled by $x = 0$ and $x=1$ in which $x=1$ occurs with probability $p$ and $x=0$ occurs with probability $q= 1-p$. The PMF $f$ of this distribution over possible outcomes, $x$, is

$$
f(k;p)=
\begin{cases}
p & {\text{if }}x=1,\\
q=1-p & {\text{if }}x=0.
\end{cases}
$$

This can also be expressed as

$$
f(k;p)=p^{x}(1-p)^{1-x}\quad {\text{for }}x\in \{0,1\}
$$

### Continuous Probability Distributions

![Untitled](images/Untitled%2017.png)

A **continuous random variable** is not defined at specific values. Instead, it is defined over an interval of values. The probability of observing any single value is equal to 0, since the number of values which may be assumed by the random variable is infinite.

Probability Density Functions

![Untitled](images/Untitled%2018.png)

A probability mass function differs from a **probability density function** (PDF) in that the latter is associated with continuous rather than discrete random variables. A PDF is represented by the **area under a curve** (integrated over an interval).

### Cumulative Distribution Function

The Cumulative Distribution Functions of a random variable $X$ shows how probability the variable has accumulated until a certain value

CDF for Discrete Probability Distributions

![Untitled](images/Untitled%2019.png)

CDF for Continuous Probability Distributions

![Untitled](images/Untitled%2020.png)

The CDF for a random variable $X$ is given by

$$
F_X(x) = P(X \leq x) = \int_{-\infty}^x f_X(t)dt
$$

![Untitled](images/Untitled%2021.png)

Every CDF $F_X$ is **non-decreasing** and **right-continuous**. Furthermore,

$$
\lim_{x\to -\infty}F_{X}(x) = 0, \quad \lim_{x\to +\infty}F_{X}(x) = 1
$$

### Uniform Distribution

A continuous random variable can be modelled with a uniform distribution if all possible values lie in an interval $[a,b]$ and have the same frequency of occurrence

![Untitled](images/Untitled%2022.png)

The probability density function of the continuous uniform distribution is:

$$
f(x)=
\begin{cases}
{\frac {1}{b-a}} & {\text{for }}a\leq x\leq b,\\
0 & {\text{for }x < a  \space {\text{ or }} \space x>b}.
\end{cases}
$$

### Normal Distribution (Gaussian Distribution)

In statistics, a **normal distribution** or **Gaussian distribution** is a type of continuous probability distribution for a real-valued random variable. In a normal distribution, data is symmetrically distributed with no skew. When plotted on a graph, the data follows a bell shape, with most values clustering around a central region and tapering off as they go further away from the center.

![Untitled](images/Untitled%2023.png)

The general form of its probability density function is

$$
f(x)={\frac {1}{\sigma {\sqrt {2\pi }}}}e^{-{\frac {1}{2}}\left({\frac {x-\mu }{\sigma }}\right)^{2}}
$$

where $\mu$ is the mean of the distribution and $\sigma$ is the standard deviation. 

The mean of a data sample ($x_1, x_2, \dots, x_n)$ is calculated by the sum of all the values divided by the number of items in the sample $N$

$$
\mu = \frac{1}{N}(\sum_{i=1}^N x_i)
$$

The standard deviation measures the amount of variation of a random variable expected around its mean. It is calculated by

$$
\sigma = \sqrt{\frac{1}{N}\sum_{i=1}^N (x_i - \mu)^2}
$$

The normal distribution is often referred to as ${\mathcal {N}}(\mu ,\sigma ^{2})$. Thus, when a random variable $X$ is normally distribution with mean $\mu$ and standard deviation $\sigma$, one may write

$$
X\sim {\mathcal {N}}(\mu ,\sigma ^{2})
$$

where $\sigma^2$ is the variance.

Empirical Rule

The empirical rule, or 68-95-99.7 rule, tells you where most of your values lie in a normal distribution

- Around 68% of values are within 1 standard deviation from the mean.
- Around 95% of values are within 2 standard deviations from the mean.
- Around 99.7% of values are within 3 standard deviations from the mean.

Standardisation

We can convert any normal distribution to the **standard normal distribution**, where $\mu =0$ and $\sigma = 1$. This is done by transforming each value $x \in X$ into its corresponding standardized $z$ value using the following formula

$$
z = \frac{x - \mu}{\sigma}
$$

Chi-Squared Distribution