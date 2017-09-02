# 20newsgroups-classification-with-Scikit-Learn
There are various algorithms which can be used for text classification.I used <a href="https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html">
Multinomial NB </a> for classification.Another methods like k-NN, SVM can be used and the differences between them can be compared.
<br><br> Converting text to vectors
<pre>
  <code>
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    tfidf_train = tfidf_vectorizer.fit_transform(twenty_train.data)
    tfidf_test = tfidf_vectorizer.transform(twenty_test.data)
  </code>
</pre>
Training data I provided with MulnomialNB:
<pre>
  <code>
    from sklearn.naive_bayes import MultinomialNB 
    clf.fit(tfidf_train, twenty_train.target)
    clf = MultinomialNB()
  </code>
</pre>
Testing performance of MultinomialNB: 
<pre>
  <code>
    from sklearn.metrics import accuracy_score,f1_score
    pred = clf.predict(tfidf_test)
    score = accuracy_score(twenty_test.target, pred)
    print("Accuracy:   %0.3f" % score)
    f1 = f1_score(twenty_test.target, pred, average='macro')
    print("F-1 Score:   %0.3f" % f1)
  </code>
</pre>
Result:
<pre>
  Accuracy:   0.817
  F-1 Score:   0.800
</pre>
Confusion Matrix:
<pre>
  <code>
  [[220   0   0   1   0   1   0   0   2   1   1   3   0   6   4  67   4   6   1   2]
  [  1 280  15  12   8  20   3   4   1   6   2  19   5   1   7   1   3   1   0   0]
  [  1  18 283  40   4  12   1   1   4   4   2  10   1   0   6   6   1   0   0   0]
  [  0   5  17 315  16   1  10   4   1   0   2   3  13   0   5   0   0   0   0   0]
  [  0   2  11  23 312   2   8   3   1   3   2   5   8   0   3   0   2   0   0   0]
  [  1  31  17  11   2 310   2   0   1   1   0  10   0   1   5   1   2   0   0   0]
  [  0   2   3  25  10   0 310  12   2   3   5   2   6   5   1   2   2   0   0   0]
  [  1   1   0   3   0   0   5 362   3   3   3   2   3   1   4   0   4   0   1   0]
  [  0   0   0   1   0   0   4   9 381   0   0   1   1   0   0   0   1   0   0   0]
  [  0   0   0   0   1   0   1   3   0 367  21   0   0   0   2   1   1   0   0   0]
  [  0   0   0   0   0   0   0   0   0   4 392   0   0   0   1   2   0   0   0   0]
  [  0   3   2   0   1   2   2   3   0   0   0 380   1   0   1   0   1   0   0   0]
  [  0   6   6  26   6   0   7   6   4   1   3  47 256   4  12   6   1   2   0   0]
  [  2   2   1   1   1   3   2   1   5   6   6   6   8 312   7  24   4   4   1   0]
  [  0   3   0   1   0   4   0   0   1   0   2   2   1   2 372   3   2   0   1   0]
  [  3   1   2   2   0   0   0   0   1   1   0   0   0   2   3 382   0   0   0   1]
  [  0   0   0   1   0   0   2   1   1   1   0   8   0   1   1   2 344   1   1   0]
  [  0   1   0   0   0   2   0   0   0   1   1   2   0   0   0   8   5 354   2   0]
  [  2   0   0   0   0   0   0   1   0   0   1   6   0   1  11  10 114   4  160  0]
  [ 43   2   1   0   0   0   0   0   0   1   2   1   0   3   6  97  29   3   4  59]]
  </code>
</pre>
