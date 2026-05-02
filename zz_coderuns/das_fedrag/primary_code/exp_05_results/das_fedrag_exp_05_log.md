# DAS-FedAvg Experiment 05 Log

**Experiment Name**: `das_fedavg_soft_domain_weighting` (8 Rounds, LR=5e-7, Pure Relevance Weighting, 3 Local Epochs)
**Target Dataset**: `nfcorpus` (medical domain)

## 1. Experiment Settings
- **Rounds**: 8
- **Local Epochs**: 3
- **Learning Rate**: 5e-07 
- **Batch Size**: 8
- **Number of Clients**: 5
- **Max Corpus Docs**: 8000
- **Max Train Pairs**: 4000
- **Max Server Val Pairs**: 400
- **Max Final Test Pairs**: 500
- **Top K**: 10
- **Domain Weighting**: Pure Relevance Proportional ($w_j = d_j / \Sigma d_k$)

## 2. Client Split Summary & Relevance Scores
| Client ID | Domain | Relevance Score | Num Training Pairs | Pair Hash |
| :--- | :--- | :--- | :--- | :--- |
| 0 | medical (TARGET) | 1.0000 | 4000 | varies by seed |
| 1 | science | 0.4791 | 239 | varies by seed |
| 4 | biomedical | 0.3212 | 101 | varies by seed |
| 3 | argument | 0.0422 | 981 | varies by seed |
| 2 | finance | 0.0130 | 168 | varies by seed |

## 3. Universal Pre-Train Metrics (Deterministic Eval Splits)
- **Pre-Server Val**: MRR: 0.0224 | Recall@10: 0.0575 | NDCG@10: 0.0304
- **Pre-Final Test**: MRR: 0.0272 | Recall@10: 0.0800 | NDCG@10: 0.0392

---

## 4. Evaluation Metrics Summary

### Baseline FedAvg (Seed 42)
- **Best Server Val**: MRR: 0.0225 | Recall@10: 0.0625 | NDCG@10: 0.0315
- **Final Test**: MRR: 0.0257 | Recall@10: 0.0740 | NDCG@10: 0.0368
- **Best Round**: 5

### Soft Domain Weighting (Seed 42)
- **Best Server Val**: MRR: 0.0225 | Recall@10: 0.0625 | NDCG@10: 0.0315
- **Final Test**: MRR: 0.0257 | Recall@10: 0.0740 | NDCG@10: 0.0369
- **Best Round**: 7

### Baseline FedAvg (Seed 123)
- **Best Server Val**: MRR: 0.0225 | Recall@10: 0.0575 | NDCG@10: 0.0305
- **Final Test**: MRR: 0.0260 | Recall@10: 0.0800 | NDCG@10: 0.0384
- **Best Round**: 8

### Soft Domain Weighting (Seed 123)
- **Best Server Val**: MRR: 0.0223 | Recall@10: 0.0625 | NDCG@10: 0.0313
- **Final Test**: MRR: 0.0264 | Recall@10: 0.0780 | NDCG@10: 0.0382
- **Best Round**: 6

### Baseline FedAvg (Seed 256)
- **Best Server Val**: MRR: 0.0236 | Recall@10: 0.0600 | NDCG@10: 0.0318
- **Final Test**: MRR: 0.0270 | Recall@10: 0.0800 | NDCG@10: 0.0391
- **Best Round**: 6

### Soft Domain Weighting (Seed 256)
- **Best Server Val**: MRR: 0.0224 | Recall@10: 0.0600 | NDCG@10: 0.0309
- **Final Test**: MRR: 0.0270 | Recall@10: 0.0800 | NDCG@10: 0.0391
- **Best Round**: 8

---

## 5. Final Comparison Table (Final Test Metrics)

| Model | Seed | Final Test MRR | Final Test Recall@10 | Final Test NDCG@10 | Best Round |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline FedAvg** | 42 | 0.0257 | 0.0740 | 0.0368 | 5 |
| **Soft Domain** | 42 | 0.0257 | 0.0740 | 0.0369 | 7 |
| **Baseline FedAvg** | 123 | 0.0260 | 0.0800 | 0.0384 | 8 |
| **Soft Domain** | 123 | 0.0264 | 0.0780 | 0.0382 | 6 |
| **Baseline FedAvg** | 256 | 0.0270 | 0.0800 | 0.0391 | 6 |
| **Soft Domain** | 256 | 0.0270 | 0.0800 | 0.0391 | 8 |


---

## 6. Detailed Round-by-Round Metrics (from CSVs)

### baseline_fedavg_seed_123_target_nfcorpus_r8_e3 - Round Metrics

| round | seed | target_dataset | avg_loss | aggregated_delta_norm | aggregated_model_hash | num_selected | num_total | best_round_so_far | selected_best_round | pre_server_val_mrr | pre_server_val_recall_at_k | pre_server_val_ndcg_at_k | pre_final_test_mrr | pre_final_test_recall_at_k | pre_final_test_ndcg_at_k | server_val_mrr | server_val_recall_at_k | server_val_ndcg_at_k | final_test_mrr | final_test_recall_at_k | final_test_ndcg_at_k | client_0_domain | client_0_relevance | client_0_selected | client_0_loss | client_0_num_examples | client_0_domain_weight | client_0_delta_norm | client_1_domain | client_1_relevance | client_1_selected | client_1_loss | client_1_num_examples | client_1_domain_weight | client_1_delta_norm | client_2_domain | client_2_relevance | client_2_selected | client_2_loss | client_2_num_examples | client_2_domain_weight | client_2_delta_norm | client_3_domain | client_3_relevance | client_3_selected | client_3_loss | client_3_num_examples | client_3_domain_weight | client_3_delta_norm | client_4_domain | client_4_relevance | client_4_selected | client_4_loss | client_4_num_examples | client_4_domain_weight | client_4_delta_norm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 123 | nfcorpus | 1.682749 | 0.086734 | ef4ffdb3bebdc86b9dd8c8ceafbe7652 | 5 | 5 | 1 | 8 | 0.022409 | 0.057500 | 0.030438 | 0.027187 | 0.080000 | 0.039210 | 0.021898 | 0.057500 | 0.029985 | 0.025967 | 0.080000 | 0.038402 | medical | 1.0000 | 1 | 2.219404 | 4000 | 0.728730 | 0.119457 | science | 0.4791 | 1 | 0.155901 | 239 | 0.043542 | 0.013365 | finance | 0.0130 | 1 | 0.259572 | 168 | 0.030607 | 0.009746 | argument | 0.0422 | 1 | 0.123954 | 981 | 0.178721 | 0.032801 | biomedical | 0.3212 | 1 | 1.549770 | 101 | 0.018400 | 0.009521 |
| 2 | 123 | nfcorpus | 1.582741 | 0.075759 | 11490b7724595be5e85a1787bbfed3d2 | 5 | 5 | 1 | 8 | 0.022409 | 0.057500 | 0.030438 | 0.027187 | 0.080000 | 0.039210 | 0.021898 | 0.057500 | 0.029985 | 0.025967 | 0.080000 | 0.038402 | medical | 1.0000 | 1 | 2.082280 | 4000 | 0.728730 | 0.104513 | science | 0.4791 | 1 | 0.164278 | 239 | 0.043542 | 0.013789 | finance | 0.0130 | 1 | 0.257895 | 168 | 0.030607 | 0.009763 | argument | 0.0422 | 1 | 0.123209 | 981 | 0.178721 | 0.033796 | biomedical | 0.3212 | 1 | 1.535567 | 101 | 0.018400 | 0.009482 |
| 3 | 123 | nfcorpus | 1.510891 | 0.067607 | a2e350f72699e027efe0bf95eeef1928 | 5 | 5 | 3 | 8 | 0.022409 | 0.057500 | 0.030438 | 0.027187 | 0.080000 | 0.039210 | 0.022027 | 0.060000 | 0.030622 | 0.025967 | 0.080000 | 0.038402 | medical | 1.0000 | 1 | 1.983567 | 4000 | 0.728730 | 0.093441 | science | 0.4791 | 1 | 0.175209 | 239 | 0.043542 | 0.014423 | finance | 0.0130 | 1 | 0.257784 | 168 | 0.030607 | 0.009792 | argument | 0.0422 | 1 | 0.122518 | 981 | 0.178721 | 0.034839 | biomedical | 0.3212 | 1 | 1.521166 | 101 | 0.018400 | 0.009465 |
| 4 | 123 | nfcorpus | 1.457480 | 0.061201 | ed7c7d673690cd47f445dddd2576a44d | 5 | 5 | 4 | 8 | 0.022409 | 0.057500 | 0.030438 | 0.027187 | 0.080000 | 0.039210 | 0.022124 | 0.062500 | 0.031205 | 0.025967 | 0.080000 | 0.038402 | medical | 1.0000 | 1 | 1.909964 | 4000 | 0.728730 | 0.084735 | science | 0.4791 | 1 | 0.187833 | 239 | 0.043542 | 0.015240 | finance | 0.0130 | 1 | 0.258594 | 168 | 0.030607 | 0.009826 | argument | 0.0422 | 1 | 0.121944 | 981 | 0.178721 | 0.035939 | biomedical | 0.3212 | 1 | 1.507809 | 101 | 0.018400 | 0.009470 |
| 5 | 123 | nfcorpus | 1.416926 | 0.055966 | ed5503e553559bce9008014a2dff9951 | 5 | 5 | 4 | 8 | 0.022409 | 0.057500 | 0.030438 | 0.027187 | 0.080000 | 0.039210 | 0.021684 | 0.057500 | 0.029817 | 0.025967 | 0.080000 | 0.038402 | medical | 1.0000 | 1 | 1.853885 | 4000 | 0.728730 | 0.077558 | science | 0.4791 | 1 | 0.200947 | 239 | 0.043542 | 0.016166 | finance | 0.0130 | 1 | 0.259799 | 168 | 0.030607 | 0.009859 | argument | 0.0422 | 1 | 0.121468 | 981 | 0.178721 | 0.037039 | biomedical | 0.3212 | 1 | 1.496385 | 101 | 0.018400 | 0.009496 |
| 6 | 123 | nfcorpus | 1.385404 | 0.051782 | f4934d7b2f04b72116b3e532aac4b57b | 5 | 5 | 4 | 8 | 0.022409 | 0.057500 | 0.030438 | 0.027187 | 0.080000 | 0.039210 | 0.021808 | 0.057500 | 0.029942 | 0.025967 | 0.080000 | 0.038402 | medical | 1.0000 | 1 | 1.810190 | 4000 | 0.728730 | 0.071716 | science | 0.4791 | 1 | 0.213251 | 239 | 0.043542 | 0.017087 | finance | 0.0130 | 1 | 0.260982 | 168 | 0.030607 | 0.009890 | argument | 0.0422 | 1 | 0.120966 | 981 | 0.178721 | 0.038022 | biomedical | 0.3212 | 1 | 1.487556 | 101 | 0.018400 | 0.009538 |
| 7 | 123 | nfcorpus | 1.360033 | 0.048624 | adc2bd68c10a5c6e34d00441617d293a | 5 | 5 | 7 | 8 | 0.022409 | 0.057500 | 0.030438 | 0.027187 | 0.080000 | 0.039210 | 0.022409 | 0.057500 | 0.030436 | 0.025967 | 0.080000 | 0.038402 | medical | 1.0000 | 1 | 1.775033 | 4000 | 0.728730 | 0.067191 | science | 0.4791 | 1 | 0.223636 | 239 | 0.043542 | 0.017920 | finance | 0.0130 | 1 | 0.261867 | 168 | 0.030607 | 0.009914 | argument | 0.0422 | 1 | 0.120289 | 981 | 0.178721 | 0.038790 | biomedical | 0.3212 | 1 | 1.481620 | 101 | 0.018400 | 0.009588 |
| 8 | 123 | nfcorpus | 1.338706 | 0.046369 | 6bd6af8a2fc9926bb8f3bb825a15420d | 5 | 5 | 8 | 8 | 0.022409 | 0.057500 | 0.030438 | 0.027187 | 0.080000 | 0.039210 | 0.022534 | 0.057500 | 0.030546 | 0.025967 | 0.080000 | 0.038402 | medical | 1.0000 | 1 | 1.745592 | 4000 | 0.728730 | 0.063857 | science | 0.4791 | 1 | 0.231470 | 239 | 0.043542 | 0.018588 | finance | 0.0130 | 1 | 0.262328 | 168 | 0.030607 | 0.009930 | argument | 0.0422 | 1 | 0.119343 | 981 | 0.178721 | 0.039268 | biomedical | 0.3212 | 1 | 1.478424 | 101 | 0.018400 | 0.009640 |


### baseline_fedavg_seed_256_target_nfcorpus_r8_e3 - Round Metrics

| round | seed | target_dataset | avg_loss | aggregated_delta_norm | aggregated_model_hash | num_selected | num_total | best_round_so_far | selected_best_round | pre_server_val_mrr | pre_server_val_recall_at_k | pre_server_val_ndcg_at_k | pre_final_test_mrr | pre_final_test_recall_at_k | pre_final_test_ndcg_at_k | server_val_mrr | server_val_recall_at_k | server_val_ndcg_at_k | final_test_mrr | final_test_recall_at_k | final_test_ndcg_at_k | client_0_domain | client_0_relevance | client_0_selected | client_0_loss | client_0_num_examples | client_0_domain_weight | client_0_delta_norm | client_1_domain | client_1_relevance | client_1_selected | client_1_loss | client_1_num_examples | client_1_domain_weight | client_1_delta_norm | client_2_domain | client_2_relevance | client_2_selected | client_2_loss | client_2_num_examples | client_2_domain_weight | client_2_delta_norm | client_3_domain | client_3_relevance | client_3_selected | client_3_loss | client_3_num_examples | client_3_domain_weight | client_3_delta_norm | client_4_domain | client_4_relevance | client_4_selected | client_4_loss | client_4_num_examples | client_4_domain_weight | client_4_delta_norm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 256 | nfcorpus | 1.678673 | 0.088044 | 3dadf80cc6063ceae8e716d51ceff297 | 5 | 5 | 1 | 6 | 0.022409 | 0.057500 | 0.030438 | 0.027187 | 0.080000 | 0.039210 | 0.021898 | 0.057500 | 0.029985 | 0.026960 | 0.080000 | 0.039110 | medical | 1.0000 | 1 | 2.209597 | 4000 | 0.728730 | 0.121284 | science | 0.4791 | 1 | 0.179353 | 239 | 0.043542 | 0.013019 | finance | 0.0130 | 1 | 0.316873 | 168 | 0.030607 | 0.009435 | argument | 0.0422 | 1 | 0.128809 | 981 | 0.178721 | 0.031467 | biomedical | 0.3212 | 1 | 1.518682 | 101 | 0.018400 | 0.009713 |
| 2 | 256 | nfcorpus | 1.575876 | 0.076412 | 45d7e4863564dde017f3c0ed8f70cfe9 | 5 | 5 | 1 | 6 | 0.022409 | 0.057500 | 0.030438 | 0.027187 | 0.080000 | 0.039210 | 0.021898 | 0.057500 | 0.029985 | 0.026960 | 0.080000 | 0.039110 | medical | 1.0000 | 1 | 2.068757 | 4000 | 0.728730 | 0.105437 | science | 0.4791 | 1 | 0.188570 | 239 | 0.043542 | 0.013467 | finance | 0.0130 | 1 | 0.314513 | 168 | 0.030607 | 0.009447 | argument | 0.0422 | 1 | 0.128193 | 981 | 0.178721 | 0.032611 | biomedical | 0.3212 | 1 | 1.497947 | 101 | 0.018400 | 0.009635 |
| 3 | 256 | nfcorpus | 1.502816 | 0.067811 | 10c969861c68d664234277a6638423d3 | 5 | 5 | 3 | 6 | 0.022409 | 0.057500 | 0.030438 | 0.027187 | 0.080000 | 0.039210 | 0.022235 | 0.060000 | 0.030795 | 0.026960 | 0.080000 | 0.039110 | medical | 1.0000 | 1 | 1.968490 | 4000 | 0.728730 | 0.093734 | science | 0.4791 | 1 | 0.200393 | 239 | 0.043542 | 0.014063 | finance | 0.0130 | 1 | 0.314194 | 168 | 0.030607 | 0.009471 | argument | 0.0422 | 1 | 0.127557 | 981 | 0.178721 | 0.033884 | biomedical | 0.3212 | 1 | 1.477082 | 101 | 0.018400 | 0.009588 |
| 4 | 256 | nfcorpus | 1.449023 | 0.061185 | 00032e0862108dd2c486c13016f025a1 | 5 | 5 | 3 | 6 | 0.022409 | 0.057500 | 0.030438 | 0.027187 | 0.080000 | 0.039210 | 0.021857 | 0.060000 | 0.030467 | 0.026960 | 0.080000 | 0.039110 | medical | 1.0000 | 1 | 1.894470 | 4000 | 0.728730 | 0.084698 | science | 0.4791 | 1 | 0.213507 | 239 | 0.043542 | 0.014827 | finance | 0.0130 | 1 | 0.315010 | 168 | 0.030607 | 0.009499 | argument | 0.0422 | 1 | 0.126957 | 981 | 0.178721 | 0.035220 | biomedical | 0.3212 | 1 | 1.458559 | 101 | 0.018400 | 0.009573 |
| 5 | 256 | nfcorpus | 1.408398 | 0.055921 | df43cd6a800116bd090d55289ebf91fe | 5 | 5 | 3 | 6 | 0.022409 | 0.057500 | 0.030438 | 0.027187 | 0.080000 | 0.039210 | 0.021978 | 0.060000 | 0.030584 | 0.026960 | 0.080000 | 0.039110 | medical | 1.0000 | 1 | 1.838397 | 4000 | 0.728730 | 0.077453 | science | 0.4791 | 1 | 0.226707 | 239 | 0.043542 | 0.015722 | finance | 0.0130 | 1 | 0.316316 | 168 | 0.030607 | 0.009527 | argument | 0.0422 | 1 | 0.126367 | 981 | 0.178721 | 0.036544 | biomedical | 0.3212 | 1 | 1.443739 | 101 | 0.018400 | 0.009583 |
| 6 | 256 | nfcorpus | 1.376833 | 0.051823 | ebbc9c77ba28833baae15c9bee64fb6d | 5 | 5 | 6 | 6 | 0.022409 | 0.057500 | 0.030438 | 0.027187 | 0.080000 | 0.039210 | 0.023645 | 0.060000 | 0.031834 | 0.026960 | 0.080000 | 0.039110 | medical | 1.0000 | 1 | 1.794733 | 4000 | 0.728730 | 0.071711 | science | 0.4791 | 1 | 0.238860 | 239 | 0.043542 | 0.016608 | finance | 0.0130 | 1 | 0.317670 | 168 | 0.030607 | 0.009552 | argument | 0.0422 | 1 | 0.125692 | 981 | 0.178721 | 0.037701 | biomedical | 0.3212 | 1 | 1.433121 | 101 | 0.018400 | 0.009612 |
| 7 | 256 | nfcorpus | 1.351338 | 0.048784 | fa4ddac94dfe4fda5917f114c5ff9ff3 | 5 | 5 | 6 | 6 | 0.022409 | 0.057500 | 0.030438 | 0.027187 | 0.080000 | 0.039210 | 0.023624 | 0.060000 | 0.031823 | 0.026960 | 0.080000 | 0.039110 | medical | 1.0000 | 1 | 1.759475 | 4000 | 0.728730 | 0.067344 | science | 0.4791 | 1 | 0.249000 | 239 | 0.043542 | 0.017394 | finance | 0.0130 | 1 | 0.318798 | 168 | 0.030607 | 0.009571 | argument | 0.0422 | 1 | 0.124820 | 981 | 0.178721 | 0.038568 | biomedical | 0.3212 | 1 | 1.426527 | 101 | 0.018400 | 0.009654 |
| 8 | 256 | nfcorpus | 1.329792 | 0.046624 | 02f91276aeca320221b5a02a1bb2ace0 | 5 | 5 | 6 | 6 | 0.022409 | 0.057500 | 0.030438 | 0.027187 | 0.080000 | 0.039210 | 0.023374 | 0.057500 | 0.031101 | 0.026960 | 0.080000 | 0.039110 | medical | 1.0000 | 1 | 1.729780 | 4000 | 0.728730 | 0.064151 | science | 0.4791 | 1 | 0.256586 | 239 | 0.043542 | 0.018018 | finance | 0.0130 | 1 | 0.319580 | 168 | 0.030607 | 0.009583 | argument | 0.0422 | 1 | 0.123687 | 981 | 0.178721 | 0.039079 | biomedical | 0.3212 | 1 | 1.423331 | 101 | 0.018400 | 0.009700 |


### baseline_fedavg_seed_42_target_nfcorpus_r8_e3 - Round Metrics

| round | seed | target_dataset | avg_loss | aggregated_delta_norm | aggregated_model_hash | num_selected | num_total | best_round_so_far | selected_best_round | pre_server_val_mrr | pre_server_val_recall_at_k | pre_server_val_ndcg_at_k | pre_final_test_mrr | pre_final_test_recall_at_k | pre_final_test_ndcg_at_k | server_val_mrr | server_val_recall_at_k | server_val_ndcg_at_k | final_test_mrr | final_test_recall_at_k | final_test_ndcg_at_k | client_0_domain | client_0_relevance | client_0_selected | client_0_loss | client_0_num_examples | client_0_domain_weight | client_0_delta_norm | client_1_domain | client_1_relevance | client_1_selected | client_1_loss | client_1_num_examples | client_1_domain_weight | client_1_delta_norm | client_2_domain | client_2_relevance | client_2_selected | client_2_loss | client_2_num_examples | client_2_domain_weight | client_2_delta_norm | client_3_domain | client_3_relevance | client_3_selected | client_3_loss | client_3_num_examples | client_3_domain_weight | client_3_delta_norm | client_4_domain | client_4_relevance | client_4_selected | client_4_loss | client_4_num_examples | client_4_domain_weight | client_4_delta_norm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 42 | nfcorpus | 1.680413 | 0.087787 | 0568f04e4f43bc70c4fed60970b015e9 | 5 | 5 | 1 | 5 | 0.022409 | 0.057500 | 0.030438 | 0.027187 | 0.080000 | 0.039210 | 0.021898 | 0.057500 | 0.029985 | 0.025715 | 0.074000 | 0.036844 | medical | 1.0000 | 1 | 2.220324 | 4000 | 0.728730 | 0.120971 | science | 0.4791 | 1 | 0.160098 | 239 | 0.043542 | 0.012950 | finance | 0.0130 | 1 | 0.279597 | 168 | 0.030607 | 0.009540 | argument | 0.0422 | 1 | 0.113319 | 981 | 0.178721 | 0.031541 | biomedical | 0.3212 | 1 | 1.446396 | 101 | 0.018400 | 0.009507 |
| 2 | 42 | nfcorpus | 1.578793 | 0.076139 | 645d35b0fbbf7b5e2dc78e0606dd94bc | 5 | 5 | 2 | 5 | 0.022409 | 0.057500 | 0.030438 | 0.027187 | 0.080000 | 0.039210 | 0.022148 | 0.060000 | 0.030708 | 0.025715 | 0.074000 | 0.036844 | medical | 1.0000 | 1 | 2.080694 | 4000 | 0.728730 | 0.105129 | science | 0.4791 | 1 | 0.168102 | 239 | 0.043542 | 0.013431 | finance | 0.0130 | 1 | 0.279042 | 168 | 0.030607 | 0.009562 | argument | 0.0422 | 1 | 0.113826 | 981 | 0.178721 | 0.032892 | biomedical | 0.3212 | 1 | 1.430680 | 101 | 0.018400 | 0.009435 |
| 3 | 42 | nfcorpus | 1.506160 | 0.067882 | 9f79d545cd1aff46e2f6c5591f080fd0 | 5 | 5 | 2 | 5 | 0.022409 | 0.057500 | 0.030438 | 0.027187 | 0.080000 | 0.039210 | 0.022124 | 0.062500 | 0.031205 | 0.025715 | 0.074000 | 0.036844 | medical | 1.0000 | 1 | 1.980606 | 4000 | 0.728730 | 0.093901 | science | 0.4791 | 1 | 0.178644 | 239 | 0.043542 | 0.014053 | finance | 0.0130 | 1 | 0.280373 | 168 | 0.030607 | 0.009590 | argument | 0.0422 | 1 | 0.114345 | 981 | 0.178721 | 0.034263 | biomedical | 0.3212 | 1 | 1.415032 | 101 | 0.018400 | 0.009397 |
| 4 | 42 | nfcorpus | 1.452160 | 0.061602 | 7079dfe2a7d24d2242b5eadb6bdf813e | 5 | 5 | 2 | 5 | 0.022409 | 0.057500 | 0.030438 | 0.027187 | 0.080000 | 0.039210 | 0.022100 | 0.062500 | 0.031185 | 0.025715 | 0.074000 | 0.036844 | medical | 1.0000 | 1 | 1.905898 | 4000 | 0.728730 | 0.085333 | science | 0.4791 | 1 | 0.190976 | 239 | 0.043542 | 0.014821 | finance | 0.0130 | 1 | 0.282684 | 168 | 0.030607 | 0.009618 | argument | 0.0422 | 1 | 0.114818 | 981 | 0.178721 | 0.035659 | biomedical | 0.3212 | 1 | 1.401445 | 101 | 0.018400 | 0.009391 |
| 5 | 42 | nfcorpus | 1.411172 | 0.056499 | 1685f046e87a0a626d4ad68659058ce4 | 5 | 5 | 5 | 5 | 0.022409 | 0.057500 | 0.030438 | 0.027187 | 0.080000 | 0.039210 | 0.022513 | 0.062500 | 0.031517 | 0.025715 | 0.074000 | 0.036844 | medical | 1.0000 | 1 | 1.848944 | 4000 | 0.728730 | 0.078298 | science | 0.4791 | 1 | 0.204036 | 239 | 0.043542 | 0.015723 | finance | 0.0130 | 1 | 0.285305 | 168 | 0.030607 | 0.009644 | argument | 0.0422 | 1 | 0.115153 | 981 | 0.178721 | 0.036991 | biomedical | 0.3212 | 1 | 1.390949 | 101 | 0.018400 | 0.009411 |
| 6 | 42 | nfcorpus | 1.379385 | 0.052393 | 07de0ad39ff1838f5a6e1d78c0a25589 | 5 | 5 | 5 | 5 | 0.022409 | 0.057500 | 0.030438 | 0.027187 | 0.080000 | 0.039210 | 0.022342 | 0.060000 | 0.030875 | 0.025715 | 0.074000 | 0.036844 | medical | 1.0000 | 1 | 1.804642 | 4000 | 0.728730 | 0.072533 | science | 0.4791 | 1 | 0.216521 | 239 | 0.043542 | 0.016698 | finance | 0.0130 | 1 | 0.287762 | 168 | 0.030607 | 0.009665 | argument | 0.0422 | 1 | 0.115220 | 981 | 0.178721 | 0.038135 | biomedical | 0.3212 | 1 | 1.383687 | 101 | 0.018400 | 0.009451 |
| 7 | 42 | nfcorpus | 1.353878 | 0.049269 | b0b53a2f1fe8ac89d7153133c4cdac5d | 5 | 5 | 5 | 5 | 0.022409 | 0.057500 | 0.030438 | 0.027187 | 0.080000 | 0.039210 | 0.022342 | 0.060000 | 0.030875 | 0.025715 | 0.074000 | 0.036844 | medical | 1.0000 | 1 | 1.769106 | 4000 | 0.728730 | 0.068034 | science | 0.4791 | 1 | 0.227225 | 239 | 0.043542 | 0.017622 | finance | 0.0130 | 1 | 0.289757 | 168 | 0.030607 | 0.009680 | argument | 0.0422 | 1 | 0.114904 | 981 | 0.178721 | 0.038971 | biomedical | 0.3212 | 1 | 1.379266 | 101 | 0.018400 | 0.009500 |
| 8 | 42 | nfcorpus | 1.332482 | 0.047018 | 1b7e07248a30dc83b433f288b66aab70 | 5 | 5 | 5 | 5 | 0.022409 | 0.057500 | 0.030438 | 0.027187 | 0.080000 | 0.039210 | 0.022342 | 0.060000 | 0.030875 | 0.025715 | 0.074000 | 0.036844 | medical | 1.0000 | 1 | 1.739438 | 4000 | 0.728730 | 0.064698 | science | 0.4791 | 1 | 0.235390 | 239 | 0.043542 | 0.018385 | finance | 0.0130 | 1 | 0.291166 | 168 | 0.030607 | 0.009688 | argument | 0.0422 | 1 | 0.114157 | 981 | 0.178721 | 0.039457 | biomedical | 0.3212 | 1 | 1.377060 | 101 | 0.018400 | 0.009551 |


### soft_domain_seed_123_target_nfcorpus_r8_e3 - Round Metrics

| round | seed | target_dataset | avg_loss | aggregated_delta_norm | aggregated_model_hash | num_selected | num_total | best_round_so_far | selected_best_round | pre_server_val_mrr | pre_server_val_recall_at_k | pre_server_val_ndcg_at_k | pre_final_test_mrr | pre_final_test_recall_at_k | pre_final_test_ndcg_at_k | server_val_mrr | server_val_recall_at_k | server_val_ndcg_at_k | final_test_mrr | final_test_recall_at_k | final_test_ndcg_at_k | client_0_domain | client_0_relevance | client_0_selected | client_0_loss | client_0_num_examples | client_0_domain_weight | client_0_delta_norm | client_1_domain | client_1_relevance | client_1_selected | client_1_loss | client_1_num_examples | client_1_domain_weight | client_1_delta_norm | client_2_domain | client_2_relevance | client_2_selected | client_2_loss | client_2_num_examples | client_2_domain_weight | client_2_delta_norm | client_3_domain | client_3_relevance | client_3_selected | client_3_loss | client_3_num_examples | client_3_domain_weight | client_3_delta_norm | client_4_domain | client_4_relevance | client_4_selected | client_4_loss | client_4_num_examples | client_4_domain_weight | client_4_delta_norm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 123 | nfcorpus | 1.682749 | 0.064083 | 0cede6e2c0f8f277ca6fb0d38c000ca4 | 5 | 5 | 1 | 6 | 0.022409 | 0.057500 | 0.030438 | 0.027187 | 0.080000 | 0.039210 | 0.022166 | 0.057500 | 0.030228 | 0.026374 | 0.078000 | 0.038222 | medical | 1.0000 | 1 | 2.219404 | 4000 | 0.538929 | 0.119457 | science | 0.4791 | 1 | 0.155901 | 239 | 0.258191 | 0.013365 | finance | 0.0130 | 1 | 0.259572 | 168 | 0.007004 | 0.009746 | argument | 0.0422 | 1 | 0.123954 | 981 | 0.022748 | 0.032801 | biomedical | 0.3212 | 1 | 1.549770 | 101 | 0.173128 | 0.009521 |
| 2 | 123 | nfcorpus | 1.606285 | 0.058104 | 7e7268d9a62c81602ea5c9b495f78c0f | 5 | 5 | 1 | 6 | 0.022409 | 0.057500 | 0.030438 | 0.027187 | 0.080000 | 0.039210 | 0.021898 | 0.057500 | 0.029985 | 0.026374 | 0.078000 | 0.038222 | medical | 1.0000 | 1 | 2.114575 | 4000 | 0.538929 | 0.108587 | science | 0.4791 | 1 | 0.159570 | 239 | 0.258191 | 0.013645 | finance | 0.0130 | 1 | 0.258564 | 168 | 0.007004 | 0.009759 | argument | 0.0422 | 1 | 0.124541 | 981 | 0.022748 | 0.033719 | biomedical | 0.3212 | 1 | 1.533147 | 101 | 0.173128 | 0.009484 |
| 3 | 123 | nfcorpus | 1.546389 | 0.053190 | bf8177aaceca3de9be798c746e8ee391 | 5 | 5 | 1 | 6 | 0.022409 | 0.057500 | 0.030438 | 0.027187 | 0.080000 | 0.039210 | 0.021898 | 0.057500 | 0.029985 | 0.026374 | 0.078000 | 0.038222 | medical | 1.0000 | 1 | 2.032336 | 4000 | 0.538929 | 0.099680 | science | 0.4791 | 1 | 0.164601 | 239 | 0.258191 | 0.014048 | finance | 0.0130 | 1 | 0.258496 | 168 | 0.007004 | 0.009780 | argument | 0.0422 | 1 | 0.125220 | 981 | 0.022748 | 0.034682 | biomedical | 0.3212 | 1 | 1.516589 | 101 | 0.173128 | 0.009459 |
| 4 | 123 | nfcorpus | 1.498641 | 0.049092 | 5aaabb19c00fffe1f9ddd72c58a1dc24 | 5 | 5 | 1 | 6 | 0.022409 | 0.057500 | 0.030438 | 0.027187 | 0.080000 | 0.039210 | 0.021999 | 0.060000 | 0.030592 | 0.026374 | 0.078000 | 0.038222 | medical | 1.0000 | 1 | 1.966635 | 4000 | 0.538929 | 0.092267 | science | 0.4791 | 1 | 0.170689 | 239 | 0.258191 | 0.014563 | finance | 0.0130 | 1 | 0.259105 | 168 | 0.007004 | 0.009806 | argument | 0.0422 | 1 | 0.126000 | 981 | 0.022748 | 0.035698 | biomedical | 0.3212 | 1 | 1.500678 | 101 | 0.173128 | 0.009446 |
| 5 | 123 | nfcorpus | 1.460129 | 0.045579 | 602954ee8cd2e8a74147047e8621385e | 5 | 5 | 1 | 6 | 0.022409 | 0.057500 | 0.030438 | 0.027187 | 0.080000 | 0.039210 | 0.021902 | 0.060000 | 0.030512 | 0.026374 | 0.078000 | 0.038222 | medical | 1.0000 | 1 | 1.913500 | 4000 | 0.538929 | 0.085898 | science | 0.4791 | 1 | 0.177432 | 239 | 0.258191 | 0.015180 | finance | 0.0130 | 1 | 0.260153 | 168 | 0.007004 | 0.009834 | argument | 0.0422 | 1 | 0.126876 | 981 | 0.022748 | 0.036760 | biomedical | 0.3212 | 1 | 1.485855 | 101 | 0.173128 | 0.009446 |
| 6 | 123 | nfcorpus | 1.428771 | 0.042534 | ee036da4104ca8da1d1497900dc3d6ee | 5 | 5 | 6 | 6 | 0.022409 | 0.057500 | 0.030438 | 0.027187 | 0.080000 | 0.039210 | 0.022267 | 0.062500 | 0.031338 | 0.026374 | 0.078000 | 0.038222 | medical | 1.0000 | 1 | 1.870111 | 4000 | 0.538929 | 0.080334 | science | 0.4791 | 1 | 0.184346 | 239 | 0.258191 | 0.015858 | finance | 0.0130 | 1 | 0.261429 | 168 | 0.007004 | 0.009862 | argument | 0.0422 | 1 | 0.127814 | 981 | 0.022748 | 0.037830 | biomedical | 0.3212 | 1 | 1.472455 | 101 | 0.173128 | 0.009457 |
| 7 | 123 | nfcorpus | 1.402935 | 0.039932 | 27b7ffe214aff97a80386a751c71b3f1 | 5 | 5 | 6 | 6 | 0.022409 | 0.057500 | 0.030438 | 0.027187 | 0.080000 | 0.039210 | 0.021684 | 0.057500 | 0.029817 | 0.026374 | 0.078000 | 0.038222 | medical | 1.0000 | 1 | 1.834275 | 4000 | 0.538929 | 0.075506 | science | 0.4791 | 1 | 0.190931 | 239 | 0.258191 | 0.016540 | finance | 0.0130 | 1 | 0.262751 | 168 | 0.007004 | 0.009889 | argument | 0.0422 | 1 | 0.128751 | 981 | 0.022748 | 0.038853 | biomedical | 0.3212 | 1 | 1.460706 | 101 | 0.173128 | 0.009477 |
| 8 | 123 | nfcorpus | 1.381282 | 0.037772 | 2de07d2305f4a35388b5942b85eeb73b | 5 | 5 | 6 | 6 | 0.022409 | 0.057500 | 0.030438 | 0.027187 | 0.080000 | 0.039210 | 0.021822 | 0.057500 | 0.029954 | 0.026374 | 0.078000 | 0.038222 | medical | 1.0000 | 1 | 1.804207 | 4000 | 0.538929 | 0.071412 | science | 0.4791 | 1 | 0.196729 | 239 | 0.258191 | 0.017211 | finance | 0.0130 | 1 | 0.263981 | 168 | 0.007004 | 0.009914 | argument | 0.0422 | 1 | 0.129601 | 981 | 0.022748 | 0.039823 | biomedical | 0.3212 | 1 | 1.450702 | 101 | 0.173128 | 0.009504 |


### soft_domain_seed_256_target_nfcorpus_r8_e3 - Round Metrics

| round | seed | target_dataset | avg_loss | aggregated_delta_norm | aggregated_model_hash | num_selected | num_total | best_round_so_far | selected_best_round | pre_server_val_mrr | pre_server_val_recall_at_k | pre_server_val_ndcg_at_k | pre_final_test_mrr | pre_final_test_recall_at_k | pre_final_test_ndcg_at_k | server_val_mrr | server_val_recall_at_k | server_val_ndcg_at_k | final_test_mrr | final_test_recall_at_k | final_test_ndcg_at_k | client_0_domain | client_0_relevance | client_0_selected | client_0_loss | client_0_num_examples | client_0_domain_weight | client_0_delta_norm | client_1_domain | client_1_relevance | client_1_selected | client_1_loss | client_1_num_examples | client_1_domain_weight | client_1_delta_norm | client_2_domain | client_2_relevance | client_2_selected | client_2_loss | client_2_num_examples | client_2_domain_weight | client_2_delta_norm | client_3_domain | client_3_relevance | client_3_selected | client_3_loss | client_3_num_examples | client_3_domain_weight | client_3_delta_norm | client_4_domain | client_4_relevance | client_4_selected | client_4_loss | client_4_num_examples | client_4_domain_weight | client_4_delta_norm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 256 | nfcorpus | 1.678673 | 0.065090 | ba9474316993f010bca196f03c781e39 | 5 | 5 | 1 | 8 | 0.022409 | 0.057500 | 0.030438 | 0.027187 | 0.080000 | 0.039210 | 0.022075 | 0.057500 | 0.030155 | 0.026960 | 0.080000 | 0.039110 | medical | 1.0000 | 1 | 2.209597 | 4000 | 0.538929 | 0.121284 | science | 0.4791 | 1 | 0.179353 | 239 | 0.258191 | 0.013019 | finance | 0.0130 | 1 | 0.316873 | 168 | 0.007004 | 0.009435 | argument | 0.0422 | 1 | 0.128809 | 981 | 0.022748 | 0.031467 | biomedical | 0.3212 | 1 | 1.518682 | 101 | 0.173128 | 0.009713 |
| 2 | 256 | nfcorpus | 1.599991 | 0.058739 | f9e2771d02ae8535a44895fd65d7eefd | 5 | 5 | 1 | 8 | 0.022409 | 0.057500 | 0.030438 | 0.027187 | 0.080000 | 0.039210 | 0.021898 | 0.057500 | 0.029985 | 0.026960 | 0.080000 | 0.039110 | medical | 1.0000 | 1 | 2.101842 | 4000 | 0.538929 | 0.109727 | science | 0.4791 | 1 | 0.183469 | 239 | 0.258191 | 0.013329 | finance | 0.0130 | 1 | 0.315204 | 168 | 0.007004 | 0.009443 | argument | 0.0422 | 1 | 0.129486 | 981 | 0.022748 | 0.032469 | biomedical | 0.3212 | 1 | 1.496582 | 101 | 0.173128 | 0.009649 |
| 3 | 256 | nfcorpus | 1.538856 | 0.053526 | 6c69c4fbbe5adf38e66d6bf49164bce3 | 5 | 5 | 1 | 8 | 0.022409 | 0.057500 | 0.030438 | 0.027187 | 0.080000 | 0.039210 | 0.021957 | 0.057500 | 0.030042 | 0.026960 | 0.080000 | 0.039110 | medical | 1.0000 | 1 | 2.018006 | 4000 | 0.538929 | 0.100271 | science | 0.4791 | 1 | 0.189068 | 239 | 0.258191 | 0.013733 | finance | 0.0130 | 1 | 0.314751 | 168 | 0.007004 | 0.009460 | argument | 0.0422 | 1 | 0.130226 | 981 | 0.022748 | 0.033603 | biomedical | 0.3212 | 1 | 1.474675 | 101 | 0.173128 | 0.009601 |
| 4 | 256 | nfcorpus | 1.490523 | 0.049222 | 93e9af72c2ad99717ba72c09f51a31eb | 5 | 5 | 4 | 8 | 0.022409 | 0.057500 | 0.030438 | 0.027187 | 0.080000 | 0.039210 | 0.022235 | 0.060000 | 0.030795 | 0.026960 | 0.080000 | 0.039110 | medical | 1.0000 | 1 | 1.951591 | 4000 | 0.538929 | 0.092463 | science | 0.4791 | 1 | 0.195625 | 239 | 0.258191 | 0.014213 | finance | 0.0130 | 1 | 0.315138 | 168 | 0.007004 | 0.009481 | argument | 0.0422 | 1 | 0.131046 | 981 | 0.022748 | 0.034794 | biomedical | 0.3212 | 1 | 1.454113 | 101 | 0.173128 | 0.009570 |
| 5 | 256 | nfcorpus | 1.451789 | 0.045589 | 1b5860f67f2b0d306aa32db45010526c | 5 | 5 | 4 | 8 | 0.022409 | 0.057500 | 0.030438 | 0.027187 | 0.080000 | 0.039210 | 0.021982 | 0.060000 | 0.030577 | 0.026960 | 0.080000 | 0.039110 | medical | 1.0000 | 1 | 1.898228 | 4000 | 0.538929 | 0.085863 | science | 0.4791 | 1 | 0.202632 | 239 | 0.258191 | 0.014790 | finance | 0.0130 | 1 | 0.316058 | 168 | 0.007004 | 0.009504 | argument | 0.0422 | 1 | 0.131935 | 981 | 0.022748 | 0.036049 | biomedical | 0.3212 | 1 | 1.435649 | 101 | 0.173128 | 0.009555 |
| 6 | 256 | nfcorpus | 1.420376 | 0.042502 | 7c3368959229f07d6194e50e0701d0c7 | 5 | 5 | 4 | 8 | 0.022409 | 0.057500 | 0.030438 | 0.027187 | 0.080000 | 0.039210 | 0.021961 | 0.060000 | 0.030569 | 0.026960 | 0.080000 | 0.039110 | medical | 1.0000 | 1 | 1.854830 | 4000 | 0.538929 | 0.080214 | science | 0.4791 | 1 | 0.209613 | 239 | 0.258191 | 0.015438 | finance | 0.0130 | 1 | 0.317263 | 168 | 0.007004 | 0.009527 | argument | 0.0422 | 1 | 0.132863 | 981 | 0.022748 | 0.037301 | biomedical | 0.3212 | 1 | 1.419696 | 101 | 0.173128 | 0.009554 |
| 7 | 256 | nfcorpus | 1.394529 | 0.039913 | 477ae1ca98048aa1f9f56b785524a80f | 5 | 5 | 4 | 8 | 0.022409 | 0.057500 | 0.030438 | 0.027187 | 0.080000 | 0.039210 | 0.021978 | 0.060000 | 0.030584 | 0.026960 | 0.080000 | 0.039110 | medical | 1.0000 | 1 | 1.819031 | 4000 | 0.538929 | 0.075412 | science | 0.4791 | 1 | 0.216112 | 239 | 0.258191 | 0.016115 | finance | 0.0130 | 1 | 0.318561 | 168 | 0.007004 | 0.009548 | argument | 0.0422 | 1 | 0.133775 | 981 | 0.022748 | 0.038469 | biomedical | 0.3212 | 1 | 1.406374 | 101 | 0.173128 | 0.009565 |
| 8 | 256 | nfcorpus | 1.372843 | 0.037805 | ff70510cd87e60688494adb16958f67a | 5 | 5 | 8 | 8 | 0.022409 | 0.057500 | 0.030438 | 0.027187 | 0.080000 | 0.039210 | 0.022395 | 0.060000 | 0.030911 | 0.026960 | 0.080000 | 0.039110 | medical | 1.0000 | 1 | 1.788954 | 4000 | 0.538929 | 0.071412 | science | 0.4791 | 1 | 0.221741 | 239 | 0.258191 | 0.016749 | finance | 0.0130 | 1 | 0.319810 | 168 | 0.007004 | 0.009567 | argument | 0.0422 | 1 | 0.134600 | 981 | 0.022748 | 0.039513 | biomedical | 0.3212 | 1 | 1.395566 | 101 | 0.173128 | 0.009585 |


### soft_domain_seed_42_target_nfcorpus_r8_e3 - Round Metrics

| round | seed | target_dataset | avg_loss | aggregated_delta_norm | aggregated_model_hash | num_selected | num_total | best_round_so_far | selected_best_round | pre_server_val_mrr | pre_server_val_recall_at_k | pre_server_val_ndcg_at_k | pre_final_test_mrr | pre_final_test_recall_at_k | pre_final_test_ndcg_at_k | server_val_mrr | server_val_recall_at_k | server_val_ndcg_at_k | final_test_mrr | final_test_recall_at_k | final_test_ndcg_at_k | client_0_domain | client_0_relevance | client_0_selected | client_0_loss | client_0_num_examples | client_0_domain_weight | client_0_delta_norm | client_1_domain | client_1_relevance | client_1_selected | client_1_loss | client_1_num_examples | client_1_domain_weight | client_1_delta_norm | client_2_domain | client_2_relevance | client_2_selected | client_2_loss | client_2_num_examples | client_2_domain_weight | client_2_delta_norm | client_3_domain | client_3_relevance | client_3_selected | client_3_loss | client_3_num_examples | client_3_domain_weight | client_3_delta_norm | client_4_domain | client_4_relevance | client_4_selected | client_4_loss | client_4_num_examples | client_4_domain_weight | client_4_delta_norm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 42 | nfcorpus | 1.680413 | 0.064930 | 944a4dae0979517a6422ad35cb288b21 | 5 | 5 | 1 | 7 | 0.022409 | 0.057500 | 0.030438 | 0.027187 | 0.080000 | 0.039210 | 0.022041 | 0.057500 | 0.030119 | 0.025743 | 0.074000 | 0.036873 | medical | 1.0000 | 1 | 2.220324 | 4000 | 0.538929 | 0.120971 | science | 0.4791 | 1 | 0.160098 | 239 | 0.258191 | 0.012950 | finance | 0.0130 | 1 | 0.279597 | 168 | 0.007004 | 0.009540 | argument | 0.0422 | 1 | 0.113319 | 981 | 0.022748 | 0.031541 | biomedical | 0.3212 | 1 | 1.446396 | 101 | 0.173128 | 0.009507 |
| 2 | 42 | nfcorpus | 1.602514 | 0.058571 | 3b0c4d37a27c6d9dbe66dc6d69df0077 | 5 | 5 | 1 | 7 | 0.022409 | 0.057500 | 0.030438 | 0.027187 | 0.080000 | 0.039210 | 0.021898 | 0.057500 | 0.029985 | 0.025743 | 0.074000 | 0.036873 | medical | 1.0000 | 1 | 2.113336 | 4000 | 0.538929 | 0.109416 | science | 0.4791 | 1 | 0.163482 | 239 | 0.258191 | 0.013281 | finance | 0.0130 | 1 | 0.279271 | 168 | 0.007004 | 0.009557 | argument | 0.0422 | 1 | 0.114792 | 981 | 0.022748 | 0.032713 | biomedical | 0.3212 | 1 | 1.428290 | 101 | 0.173128 | 0.009443 |
| 3 | 42 | nfcorpus | 1.541798 | 0.053492 | 55717b8a80f047909ef4b98a3861a1eb | 5 | 5 | 1 | 7 | 0.022409 | 0.057500 | 0.030438 | 0.027187 | 0.080000 | 0.039210 | 0.021939 | 0.060000 | 0.030534 | 0.025743 | 0.074000 | 0.036873 | medical | 1.0000 | 1 | 2.029753 | 4000 | 0.538929 | 0.100224 | science | 0.4791 | 1 | 0.168221 | 239 | 0.258191 | 0.013699 | finance | 0.0130 | 1 | 0.280088 | 168 | 0.007004 | 0.009578 | argument | 0.0422 | 1 | 0.116401 | 981 | 0.022748 | 0.033927 | biomedical | 0.3212 | 1 | 1.410569 | 101 | 0.173128 | 0.009400 |
| 4 | 42 | nfcorpus | 1.493478 | 0.049377 | 8452292c2aee80967daa8f63f28b5d2c | 5 | 5 | 4 | 7 | 0.022409 | 0.057500 | 0.030438 | 0.027187 | 0.080000 | 0.039210 | 0.022267 | 0.062500 | 0.031338 | 0.025743 | 0.074000 | 0.036873 | medical | 1.0000 | 1 | 1.963028 | 4000 | 0.538929 | 0.092785 | science | 0.4791 | 1 | 0.174065 | 239 | 0.258191 | 0.014197 | finance | 0.0130 | 1 | 0.281672 | 168 | 0.007004 | 0.009600 | argument | 0.0422 | 1 | 0.118104 | 981 | 0.022748 | 0.035199 | biomedical | 0.3212 | 1 | 1.394113 | 101 | 0.173128 | 0.009374 |
| 5 | 42 | nfcorpus | 1.454545 | 0.045899 | 97a760d53dc7574565ec3f2cbc32f40a | 5 | 5 | 4 | 7 | 0.022409 | 0.057500 | 0.030438 | 0.027187 | 0.080000 | 0.039210 | 0.022184 | 0.062500 | 0.031262 | 0.025743 | 0.074000 | 0.036873 | medical | 1.0000 | 1 | 1.909063 | 4000 | 0.538929 | 0.086483 | science | 0.4791 | 1 | 0.180677 | 239 | 0.258191 | 0.014779 | finance | 0.0130 | 1 | 0.283707 | 168 | 0.007004 | 0.009623 | argument | 0.0422 | 1 | 0.119848 | 981 | 0.022748 | 0.036489 | biomedical | 0.3212 | 1 | 1.379516 | 101 | 0.173128 | 0.009365 |
| 6 | 42 | nfcorpus | 1.422902 | 0.042894 | 2312dd570f4948ca6c1955a7a87c294d | 5 | 5 | 4 | 7 | 0.022409 | 0.057500 | 0.030438 | 0.027187 | 0.080000 | 0.039210 | 0.022180 | 0.062500 | 0.031266 | 0.025743 | 0.074000 | 0.036873 | medical | 1.0000 | 1 | 1.865029 | 4000 | 0.538929 | 0.080994 | science | 0.4791 | 1 | 0.187613 | 239 | 0.258191 | 0.015436 | finance | 0.0130 | 1 | 0.285935 | 168 | 0.007004 | 0.009645 | argument | 0.0422 | 1 | 0.121559 | 981 | 0.022748 | 0.037762 | biomedical | 0.3212 | 1 | 1.367020 | 101 | 0.173128 | 0.009371 |
| 7 | 42 | nfcorpus | 1.396904 | 0.040317 | a0d74b7ff0e0a63b5320d26280020a26 | 5 | 5 | 7 | 7 | 0.022409 | 0.057500 | 0.030438 | 0.027187 | 0.080000 | 0.039210 | 0.022513 | 0.062500 | 0.031517 | 0.025743 | 0.074000 | 0.036873 | medical | 1.0000 | 1 | 1.828728 | 4000 | 0.538929 | 0.076221 | science | 0.4791 | 1 | 0.194359 | 239 | 0.258191 | 0.016139 | finance | 0.0130 | 1 | 0.288151 | 168 | 0.007004 | 0.009666 | argument | 0.0422 | 1 | 0.123156 | 981 | 0.022748 | 0.038951 | biomedical | 0.3212 | 1 | 1.356601 | 101 | 0.173128 | 0.009389 |
| 8 | 42 | nfcorpus | 1.375185 | 0.038172 | 61bb6e4abce792590d0568a0c19edf7b | 5 | 5 | 7 | 7 | 0.022409 | 0.057500 | 0.030438 | 0.027187 | 0.080000 | 0.039210 | 0.022308 | 0.060000 | 0.030839 | 0.025743 | 0.074000 | 0.036873 | medical | 1.0000 | 1 | 1.798349 | 4000 | 0.538929 | 0.072165 | science | 0.4791 | 1 | 0.200406 | 239 | 0.258191 | 0.016848 | finance | 0.0130 | 1 | 0.290201 | 168 | 0.007004 | 0.009683 | argument | 0.0422 | 1 | 0.124557 | 981 | 0.022748 | 0.040022 | biomedical | 0.3212 | 1 | 1.348053 | 101 | 0.173128 | 0.009415 |




---

## 7. Acceptance Reports

### baseline_fedavg_seed_123_target_nfcorpus_r8_e3 - Acceptance Report

```json
{
  "distinct_round_trajectories": {
    "applicable": true,
    "evidence": {
      "rounds": 8,
      "unique_hashes": 8
    },
    "passed": true
  },
  "high_relevance_outweighs_low_relevance": {
    "applicable": true,
    "evidence": {
      "high_cid": "0",
      "low_cid": "2",
      "per_round": [
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true
      ]
    },
    "passed": true
  },
  "medical_client_dominates": {
    "applicable": true,
    "evidence": {
      "per_round": [
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true
      ],
      "target_cid": "0"
    },
    "passed": true
  },
  "relevance_scores_ordered": {
    "applicable": true,
    "evidence": {
      "scores": {
        "0": "1.0000",
        "1": "0.4791",
        "2": "0.0130",
        "3": "0.0422",
        "4": "0.3212"
      }
    },
    "passed": true
  },
  "target_client_always_selected": {
    "applicable": true,
    "passed": true
  }
}
```

### baseline_fedavg_seed_256_target_nfcorpus_r8_e3 - Acceptance Report

```json
{
  "distinct_round_trajectories": {
    "applicable": true,
    "evidence": {
      "rounds": 8,
      "unique_hashes": 8
    },
    "passed": true
  },
  "high_relevance_outweighs_low_relevance": {
    "applicable": true,
    "evidence": {
      "high_cid": "0",
      "low_cid": "2",
      "per_round": [
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true
      ]
    },
    "passed": true
  },
  "medical_client_dominates": {
    "applicable": true,
    "evidence": {
      "per_round": [
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true
      ],
      "target_cid": "0"
    },
    "passed": true
  },
  "relevance_scores_ordered": {
    "applicable": true,
    "evidence": {
      "scores": {
        "0": "1.0000",
        "1": "0.4791",
        "2": "0.0130",
        "3": "0.0422",
        "4": "0.3212"
      }
    },
    "passed": true
  },
  "target_client_always_selected": {
    "applicable": true,
    "passed": true
  }
}
```

### baseline_fedavg_seed_42_target_nfcorpus_r8_e3 - Acceptance Report

```json
{
  "distinct_round_trajectories": {
    "applicable": true,
    "evidence": {
      "rounds": 8,
      "unique_hashes": 8
    },
    "passed": true
  },
  "high_relevance_outweighs_low_relevance": {
    "applicable": true,
    "evidence": {
      "high_cid": "0",
      "low_cid": "2",
      "per_round": [
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true
      ]
    },
    "passed": true
  },
  "medical_client_dominates": {
    "applicable": true,
    "evidence": {
      "per_round": [
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true
      ],
      "target_cid": "0"
    },
    "passed": true
  },
  "relevance_scores_ordered": {
    "applicable": true,
    "evidence": {
      "scores": {
        "0": "1.0000",
        "1": "0.4791",
        "2": "0.0130",
        "3": "0.0422",
        "4": "0.3212"
      }
    },
    "passed": true
  },
  "target_client_always_selected": {
    "applicable": true,
    "passed": true
  }
}
```

### soft_domain_seed_123_target_nfcorpus_r8_e3 - Acceptance Report

```json
{
  "distinct_round_trajectories": {
    "applicable": true,
    "evidence": {
      "rounds": 8,
      "unique_hashes": 8
    },
    "passed": true
  },
  "high_relevance_outweighs_low_relevance": {
    "applicable": true,
    "evidence": {
      "high_cid": "0",
      "low_cid": "2",
      "per_round": [
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true
      ]
    },
    "passed": true
  },
  "medical_client_dominates": {
    "applicable": true,
    "evidence": {
      "per_round": [
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true
      ],
      "target_cid": "0"
    },
    "passed": true
  },
  "relevance_scores_ordered": {
    "applicable": true,
    "evidence": {
      "scores": {
        "0": "1.0000",
        "1": "0.4791",
        "2": "0.0130",
        "3": "0.0422",
        "4": "0.3212"
      }
    },
    "passed": true
  },
  "target_client_always_selected": {
    "applicable": true,
    "passed": true
  }
}
```

### soft_domain_seed_256_target_nfcorpus_r8_e3 - Acceptance Report

```json
{
  "distinct_round_trajectories": {
    "applicable": true,
    "evidence": {
      "rounds": 8,
      "unique_hashes": 8
    },
    "passed": true
  },
  "high_relevance_outweighs_low_relevance": {
    "applicable": true,
    "evidence": {
      "high_cid": "0",
      "low_cid": "2",
      "per_round": [
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true
      ]
    },
    "passed": true
  },
  "medical_client_dominates": {
    "applicable": true,
    "evidence": {
      "per_round": [
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true
      ],
      "target_cid": "0"
    },
    "passed": true
  },
  "relevance_scores_ordered": {
    "applicable": true,
    "evidence": {
      "scores": {
        "0": "1.0000",
        "1": "0.4791",
        "2": "0.0130",
        "3": "0.0422",
        "4": "0.3212"
      }
    },
    "passed": true
  },
  "target_client_always_selected": {
    "applicable": true,
    "passed": true
  }
}
```

### soft_domain_seed_42_target_nfcorpus_r8_e3 - Acceptance Report

```json
{
  "distinct_round_trajectories": {
    "applicable": true,
    "evidence": {
      "rounds": 8,
      "unique_hashes": 8
    },
    "passed": true
  },
  "high_relevance_outweighs_low_relevance": {
    "applicable": true,
    "evidence": {
      "high_cid": "0",
      "low_cid": "2",
      "per_round": [
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true
      ]
    },
    "passed": true
  },
  "medical_client_dominates": {
    "applicable": true,
    "evidence": {
      "per_round": [
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true
      ],
      "target_cid": "0"
    },
    "passed": true
  },
  "relevance_scores_ordered": {
    "applicable": true,
    "evidence": {
      "scores": {
        "0": "1.0000",
        "1": "0.4791",
        "2": "0.0130",
        "3": "0.0422",
        "4": "0.3212"
      }
    },
    "passed": true
  },
  "target_client_always_selected": {
    "applicable": true,
    "passed": true
  }
}
```

