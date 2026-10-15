# Labeling guide for ISBA judge evaluations

Label each judge from the evaluation text in the sheet **only**. Don't use outside knowledge, even
when you know the answer is different. These labels test what can be read from the text. Put
outside knowledge in a separate `notes` column if you want to keep it.

Use `1` when the text clearly says it and `0` otherwise, including when the text doesn't say
either way ("sole practitioner" with no type of practice listed is a `0`).

| Flag | 1 when the text says the judge... | Not enough on its own |
|---|---|---|
| `former_prosecutor` | worked as an Assistant State's Attorney or otherwise prosecuted cases before becoming a judge | the State's Attorney's **Civil Division**; city or agency law departments |
| `former_criminal_defense` | represented people accused of crimes: public defender, appellate defender, private criminal defense | civil or insurance defense; "private practice" with no type listed |
| `represented_vulnerable_clients` | worked on behalf of children, poor people, or people with disabilities: public defender, public guardian, child representative, guardian ad litem, legal aid | general private practice |
| `criminal_bench` | presided over criminal matters as a judge: Criminal Division, felony or misdemeanor courtrooms, preliminary hearings, Domestic Violence court, Juvenile Justice (delinquency), bond court | **Traffic court**; a municipal district or division named without saying it was criminal ("First Municipal", "Fourth Municipal District", "County Division"); Child Protection |
| `treatment_court` | presided over mental health, drug, veterans, or other problem-solving courts | narcotics courts; domestic violence courts; after-hours or emergency duty |
| `has_criticism` | is criticized by attorneys or interviewees in any way, even in an otherwise positive evaluation ("Despite comments about poor temperament...") | the bar's overall recommendation alone |

Why these definitions: the goal is judging whether a judge is fair and unbiased in sentencing.
Career background is context for that, so each flag asks what experience the judge has with
people accused of crimes, and with vulnerable people. Sentencing data and attorney feedback are
the stronger evidence; see the judge profiles.
