# The writing standard: ASD-STE100 Simplified Technical English

Use these rules for every README and for `docs/ste-style-guide.md` in each repository. Copy this file
into the repository as `docs/ste-style-guide.md` and add a **project vocabulary** section (Section 3)
with the technical names and technical verbs of that project.

## 1. The writing rules

### Words

1. Use one word for one meaning, and one meaning for one word. Do not use synonyms for variety.
2. Use a word only as one part of speech. For example, `test` is a noun or a verb, `check` is a verb.
3. Do not use phrasal verbs (`set up`, `carry out`, `find out`, `pick up`, `look up`, `come up with`).
   Use one verb: `prepare`, `do`, `find`, `get`, `make`.
4. Do not use an `-ing` form as a noun or an adjective (`the running job`, `after indexing`).
   Exception: a technical name, a file name, a command or a status value.
5. Do not use contractions (`don't`, `it's`, `can't`). Do not use slang or idioms
   (`out of the box`, `under the hood`, `at a glance`, `gotcha`, `bells and whistles`).
6. Do not use `and/or`. Write `A, B or both`.
7. Do not use `should`, `could`, `would` or `may` for instructions. Use `must` for a rule, the
   imperative for a step and `can` for a possibility.
8. Keep the articles `a`, `an` and `the` in sentences.
9. Do not make a noun cluster of more than three words. A technical name is one word.

### Sentences

1. A procedural sentence (an instruction) has a maximum of **20 words**.
2. A descriptive sentence has a maximum of **25 words**.
3. Write one instruction in one sentence.
4. Use the imperative for an instruction: `Run the tests.` Not `The tests should be run.`
5. Use the active voice. Use the passive voice only when the agent of the action is not important.
6. Use only the simple present, the simple past and the simple future.
7. Put a condition before the instruction: `If the index is stale, build it again.`
8. Do not use semicolons in sentences. Write two sentences.

### Paragraphs, notes and warnings

1. A paragraph has one topic and a maximum of **6 sentences**. Start with the topic sentence.
2. A warning or a caution starts with a clear command. Then it gives the reason.
3. A note gives information. It does not give an instruction.
4. Use a vertical list for a sequence or a set of conditions. Each item of a numbered procedure is one step.

### Tables, headings and diagrams

1. A table cell can be a short phrase. If a cell has a sentence, the sentence obeys the rules.
2. A heading is a noun phrase (`The cost model`) or an imperative (`Run the demo`).
   Do not start a heading with an `-ing` form.
3. A diagram label is a short phrase. Use the same terms as the text.

### What STE does not change

Code, commands, file names, paths, field names, environment variables, status values, enum values,
product names and URLs stay exactly as they are. They are technical names. Put them in backticks.

## 2. General words to replace

| Do not use | Use |
|---|---|
| utilize, leverage | use |
| in order to | to |
| set up | prepare, install, configure |
| carry out, perform | do |
| make sure, ensure | make sure (allowed), or `check that` |
| a lot of, lots of | many, much |
| e.g., i.e. | for example, that is |
| should (instruction) | must (rule) / imperative (step) |
| might, may (possibility) | can |
| very, really, just, simply, easily | (delete) |
| seamless, robust, powerful, blazing | (delete or give a measured fact) |

## 3. Project vocabulary

This section gives the technical names and the technical verbs of crash-whatif. The README uses each term with only this meaning.

### 3.1 Technical names (nouns)

| Term | Meaning | Do not use |
|---|---|---|
| **crash** | One row of the crash table | accident, incident, record |
| **severity** | The label: `minor`, `moderate` or `severe` | outcome, class (alone) |
| **severe crash** | A crash with severity `severe` (binary target 1) | serious crash, bad crash |
| **flag** | A 0/1 safety column: `abs`, `esc`, `tcs`, `tpms` | boolean, indicator |
| **nominal column** | A category column with no order, for example `make` | label-encoded column |
| **pipeline** | Imputer, optional resampler, encoder and classifier as one object | workflow, chain |
| **baseline** | The `majority` model or the `make_only` model | benchmark, dummy |
| **threshold** | The probability cut that makes a prediction, chosen on the validation part | cut-off, decision boundary |
| **validation part** | The rows that choose the threshold | dev set, tuning set |
| **test part** | The real rows for the final metrics, never resampled | holdout set, eval set |
| **permutation importance** | The drop of ROC AUC when one input column is shuffled | feature importance (alone), weight |
| **counterfactual** | A changed copy of a crash that the model predicts as not severe | what-if row, alternative |
| **query** | The crash that a counterfactual explains | instance, example |
| **actionable column** | A column that a counterfactual can change | mutable feature, free feature |
| **permitted range** | The training minimum and maximum of an actionable column | bound, limit |
| **validity, proximity, sparsity, diversity, plausibility** | The five counterfactual metrics in section 9 | quality, score (alone) |
| **signal audit** | The comparison of `make` alone with all other columns | data check, sanity check |
| **generator** | `RandomSearchCF` or `DiceCF` | explainer, engine |

### 3.2 Technical verbs

| Verb | Meaning |
|---|---|
| **validate** | Check the crash table against the schema and clean each column |
| **split** | Cut the rows into train, validation and test parts with the class shares |
| **fit** | Learn the pipeline parameters from the training part |
| **resample** | Add minority rows to the training part inside the pipeline |
| **shuffle** | Permute the values of one column in the test part |
| **generate** | Make counterfactuals for one query |
| **sparsify** | Put back each changed value that is not necessary |
| **audit** | Measure how much `make` alone explains |
