"""
Collector for great/seminal research papers in AI/ML.
"""

import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.collectors.semantic_scholar_client import SemanticScholarClient
from src.storage.database import PaperDatabase


class GreatPapersCollector:
    """Collector for curated list of seminal AI/ML research papers."""

    # Seminal papers in AI/ML (100+ papers)
    SEMINAL_PAPERS = [
        # === Transformers & Attention ===
        {"title": "Attention Is All You Need", "arxiv_id": "1706.03762", "year": 2017},
        {
            "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
            "arxiv_id": "1810.04805",
            "year": 2018,
        },
        {"title": "Language Models are Unsupervised Multitask Learners", "year": 2019},  # GPT-2
        {
            "title": "Language Models are Few-Shot Learners",
            "arxiv_id": "2005.14165",
            "year": 2020,
        },  # GPT-3
        {
            "title": "Training language models to follow instructions with human feedback",
            "arxiv_id": "2203.02155",
            "year": 2022,
        },  # InstructGPT
        {
            "title": "Constitutional AI: Harmlessness from AI Feedback",
            "arxiv_id": "2212.08073",
            "year": 2022,
        },
        {
            "title": "LLaMA: Open and Efficient Foundation Language Models",
            "arxiv_id": "2302.13971",
            "year": 2023,
        },
        {
            "title": "Llama 2: Open Foundation and Fine-Tuned Chat Models",
            "arxiv_id": "2307.09288",
            "year": 2023,
        },
        # === Deep Learning Foundations ===
        {
            "title": "ImageNet Classification with Deep Convolutional Neural Networks",
            "year": 2012,
        },  # AlexNet
        {
            "title": "Deep Residual Learning for Image Recognition",
            "arxiv_id": "1512.03385",
            "year": 2015,
        },  # ResNet
        {
            "title": "Very Deep Convolutional Networks for Large-Scale Image Recognition",
            "arxiv_id": "1409.1556",
            "year": 2014,
        },  # VGGNet
        {
            "title": "Going Deeper with Convolutions",
            "arxiv_id": "1409.4842",
            "year": 2014,
        },  # GoogLeNet
        {"title": "Network In Network", "arxiv_id": "1312.4400", "year": 2013},
        {
            "title": "Densely Connected Convolutional Networks",
            "arxiv_id": "1608.06993",
            "year": 2016,
        },  # DenseNet
        {
            "title": "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications",
            "arxiv_id": "1704.04861",
            "year": 2017,
        },
        {
            "title": "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks",
            "arxiv_id": "1905.11946",
            "year": 2019,
        },
        # === Generative Models ===
        {"title": "Generative Adversarial Networks", "arxiv_id": "1406.2661", "year": 2014},
        {"title": "Conditional Generative Adversarial Nets", "arxiv_id": "1411.1784", "year": 2014},
        {
            "title": "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks",
            "arxiv_id": "1511.06434",
            "year": 2015,
        },  # DCGAN
        {
            "title": "Progressive Growing of GANs for Improved Quality, Stability, and Variation",
            "arxiv_id": "1710.10196",
            "year": 2017,
        },
        {
            "title": "A Style-Based Generator Architecture for Generative Adversarial Networks",
            "arxiv_id": "1812.04948",
            "year": 2018,
        },  # StyleGAN
        {
            "title": "Denoising Diffusion Probabilistic Models",
            "arxiv_id": "2006.11239",
            "year": 2020,
        },
        {"title": "Denoising Diffusion Implicit Models", "arxiv_id": "2010.02502", "year": 2020},
        {
            "title": "High-Resolution Image Synthesis with Latent Diffusion Models",
            "arxiv_id": "2112.10752",
            "year": 2021,
        },  # Stable Diffusion
        {"title": "Auto-Encoding Variational Bayes", "arxiv_id": "1312.6114", "year": 2013},  # VAE
        # === Computer Vision ===
        {
            "title": "You Only Look Once: Unified, Real-Time Object Detection",
            "arxiv_id": "1506.02640",
            "year": 2015,
        },  # YOLO
        {
            "title": "Rich feature hierarchies for accurate object detection and semantic segmentation",
            "arxiv_id": "1311.2524",
            "year": 2013,
        },  # R-CNN
        {"title": "Fast R-CNN", "arxiv_id": "1504.08083", "year": 2015},
        {
            "title": "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks",
            "arxiv_id": "1506.01497",
            "year": 2015,
        },
        {"title": "Mask R-CNN", "arxiv_id": "1703.06870", "year": 2017},
        {
            "title": "Fully Convolutional Networks for Semantic Segmentation",
            "arxiv_id": "1411.4038",
            "year": 2014,
        },
        {
            "title": "U-Net: Convolutional Networks for Biomedical Image Segmentation",
            "arxiv_id": "1505.04597",
            "year": 2015,
        },
        {
            "title": "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
            "arxiv_id": "2010.11929",
            "year": 2020,
        },  # Vision Transformer
        {
            "title": "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows",
            "arxiv_id": "2103.14030",
            "year": 2021,
        },
        # === NLP Classics ===
        {
            "title": "Efficient Estimation of Word Representations in Vector Space",
            "arxiv_id": "1301.3781",
            "year": 2013,
        },  # Word2Vec
        {"title": "GloVe: Global Vectors for Word Representation", "year": 2014},
        {
            "title": "Sequence to Sequence Learning with Neural Networks",
            "arxiv_id": "1409.3215",
            "year": 2014,
        },
        {
            "title": "Neural Machine Translation by Jointly Learning to Align and Translate",
            "arxiv_id": "1409.0473",
            "year": 2014,
        },  # Attention mechanism
        {"title": "Long Short-Term Memory", "year": 1997},  # LSTM
        {
            "title": "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation",
            "arxiv_id": "1406.1078",
            "year": 2014,
        },  # GRU
        {
            "title": "ELMo: Deep contextualized word representations",
            "arxiv_id": "1802.05365",
            "year": 2018,
        },
        {
            "title": "Universal Language Model Fine-tuning for Text Classification",
            "arxiv_id": "1801.06146",
            "year": 2018,
        },  # ULMFiT
        {
            "title": "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
            "arxiv_id": "1907.11692",
            "year": 2019,
        },
        {
            "title": "ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators",
            "arxiv_id": "2003.10555",
            "year": 2020,
        },
        {
            "title": "T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer",
            "arxiv_id": "1910.10683",
            "year": 2019,
        },
        # === Reinforcement Learning ===
        {
            "title": "Playing Atari with Deep Reinforcement Learning",
            "arxiv_id": "1312.5602",
            "year": 2013,
        },  # DQN
        {
            "title": "Human-level control through deep reinforcement learning",
            "year": 2015,
        },  # DQN Nature
        {
            "title": "Mastering the game of Go with deep neural networks and tree search",
            "year": 2016,
        },  # AlphaGo
        {"title": "Mastering the game of Go without human knowledge", "year": 2017},  # AlphaGo Zero
        {
            "title": "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm",
            "arxiv_id": "1712.01815",
            "year": 2017,
        },  # AlphaZero
        {
            "title": "Proximal Policy Optimization Algorithms",
            "arxiv_id": "1707.06347",
            "year": 2017,
        },  # PPO
        {
            "title": "Asynchronous Methods for Deep Reinforcement Learning",
            "arxiv_id": "1602.01783",
            "year": 2016,
        },  # A3C
        {
            "title": "Continuous control with deep reinforcement learning",
            "arxiv_id": "1509.02971",
            "year": 2015,
        },  # DDPG
        {
            "title": "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor",
            "arxiv_id": "1801.01290",
            "year": 2018,
        },
        {
            "title": "Trust Region Policy Optimization",
            "arxiv_id": "1502.05477",
            "year": 2015,
        },  # TRPO
        # === Optimization & Training ===
        {
            "title": "Adam: A Method for Stochastic Optimization",
            "arxiv_id": "1412.6980",
            "year": 2014,
        },
        {
            "title": "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift",
            "arxiv_id": "1502.03167",
            "year": 2015,
        },
        {
            "title": "Dropout: A Simple Way to Prevent Neural Networks from Overfitting",
            "year": 2014,
        },
        {"title": "Layer Normalization", "arxiv_id": "1607.06450", "year": 2016},
        {"title": "Group Normalization", "arxiv_id": "1803.08494", "year": 2018},
        {
            "title": "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification",
            "arxiv_id": "1502.01852",
            "year": 2015,
        },  # He initialization
        {
            "title": "Understanding the difficulty of training deep feedforward neural networks",
            "year": 2010,
        },  # Xavier initialization
        {
            "title": "On the importance of initialization and momentum in deep learning",
            "year": 2013,
        },
        {
            "title": "Improving neural networks by preventing co-adaptation of feature detectors",
            "arxiv_id": "1207.0580",
            "year": 2012,
        },  # Dropout original
        {
            "title": "Deep Residual Learning for Image Recognition",
            "arxiv_id": "1512.03385",
            "year": 2015,
        },
        {
            "title": "Identity Mappings in Deep Residual Networks",
            "arxiv_id": "1603.05027",
            "year": 2016,
        },
        # === Meta-Learning & Few-Shot ===
        {
            "title": "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks",
            "arxiv_id": "1703.03400",
            "year": 2017,
        },  # MAML
        {
            "title": "Prototypical Networks for Few-shot Learning",
            "arxiv_id": "1703.05175",
            "year": 2017,
        },
        {
            "title": "Matching Networks for One Shot Learning",
            "arxiv_id": "1606.04080",
            "year": 2016,
        },
        {
            "title": "Learning to Compare: Relation Network for Few-Shot Learning",
            "arxiv_id": "1711.06025",
            "year": 2017,
        },
        # === Self-Supervised Learning ===
        {
            "title": "Momentum Contrast for Unsupervised Visual Representation Learning",
            "arxiv_id": "1911.05722",
            "year": 2019,
        },  # MoCo
        {
            "title": "A Simple Framework for Contrastive Learning of Visual Representations",
            "arxiv_id": "2002.05709",
            "year": 2020,
        },  # SimCLR
        {
            "title": "Bootstrap your own latent: A new approach to self-supervised Learning",
            "arxiv_id": "2006.07733",
            "year": 2020,
        },  # BYOL
        {
            "title": "Masked Autoencoders Are Scalable Vision Learners",
            "arxiv_id": "2111.06377",
            "year": 2021,
        },  # MAE
        # === Graph Neural Networks ===
        {
            "title": "Semi-Supervised Classification with Graph Convolutional Networks",
            "arxiv_id": "1609.02907",
            "year": 2016,
        },  # GCN
        {"title": "Graph Attention Networks", "arxiv_id": "1710.10903", "year": 2017},  # GAT
        {
            "title": "Inductive Representation Learning on Large Graphs",
            "arxiv_id": "1706.02216",
            "year": 2017,
        },  # GraphSAGE
        {
            "title": "How Powerful are Graph Neural Networks?",
            "arxiv_id": "1810.00826",
            "year": 2018,
        },  # GIN
        # === Multimodal Learning ===
        {
            "title": "Learning Transferable Visual Models From Natural Language Supervision",
            "arxiv_id": "2103.00020",
            "year": 2021,
        },  # CLIP
        {
            "title": "ALIGN: Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision",
            "arxiv_id": "2102.05918",
            "year": 2021,
        },
        {
            "title": "Flamingo: a Visual Language Model for Few-Shot Learning",
            "arxiv_id": "2204.14198",
            "year": 2022,
        },
        {
            "title": "BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation",
            "arxiv_id": "2201.12086",
            "year": 2022,
        },
        # === Neural Architecture Search ===
        {
            "title": "Neural Architecture Search with Reinforcement Learning",
            "arxiv_id": "1611.01578",
            "year": 2016,
        },
        {
            "title": "Efficient Neural Architecture Search via Parameter Sharing",
            "arxiv_id": "1802.03268",
            "year": 2018,
        },  # ENAS
        {
            "title": "DARTS: Differentiable Architecture Search",
            "arxiv_id": "1806.09055",
            "year": 2018,
        },
        # === Interpretability & Explainability ===
        {
            "title": "Visualizing and Understanding Convolutional Networks",
            "arxiv_id": "1311.2901",
            "year": 2013,
        },
        {
            "title": "Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps",
            "arxiv_id": "1312.6034",
            "year": 2013,
        },
        {
            "title": "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization",
            "arxiv_id": "1610.02391",
            "year": 2016,
        },
        {
            "title": "A Unified Approach to Interpreting Model Predictions",
            "arxiv_id": "1705.07874",
            "year": 2017,
        },  # SHAP
        {
            "title": "Axiomatic Attribution for Deep Networks",
            "arxiv_id": "1703.01365",
            "year": 2017,
        },  # Integrated Gradients
        # === Adversarial Robustness ===
        {
            "title": "Intriguing properties of neural networks",
            "arxiv_id": "1312.6199",
            "year": 2013,
        },
        {
            "title": "Explaining and Harnessing Adversarial Examples",
            "arxiv_id": "1412.6572",
            "year": 2014,
        },  # FGSM
        {
            "title": "Towards Deep Learning Models Resistant to Adversarial Attacks",
            "arxiv_id": "1706.06083",
            "year": 2017,
        },  # PGD
        {"title": "Adversarial Training for Free!", "arxiv_id": "1904.12843", "year": 2019},
        # === Classic ML Papers ===
        {"title": "Random Forests", "year": 2001},
        {
            "title": "XGBoost: A Scalable Tree Boosting System",
            "arxiv_id": "1603.02754",
            "year": 2016,
        },
        {"title": "Support Vector Networks", "year": 1995},  # SVM
        {
            "title": "A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting",
            "year": 1995,
        },  # AdaBoost
        # === Theoretical Foundations ===
        {"title": "Deep Learning", "year": 2015},  # Goodfellow et al. book
        {
            "title": "Understanding deep learning requires rethinking generalization",
            "arxiv_id": "1611.03530",
            "year": 2016,
        },
        {
            "title": "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks",
            "arxiv_id": "1803.03635",
            "year": 2018,
        },
        {
            "title": "Neural Ordinary Differential Equations",
            "arxiv_id": "1806.07366",
            "year": 2018,
        },  # Neural ODEs
        {
            "title": "On the Measure of Intelligence",
            "arxiv_id": "1911.01547",
            "year": 2019,
        },  # François Chollet
        # === Scaling & Large Models ===
        {
            "title": "Scaling Laws for Neural Language Models",
            "arxiv_id": "2001.08361",
            "year": 2020,
        },
        {
            "title": "Training Compute-Optimal Large Language Models",
            "arxiv_id": "2203.15556",
            "year": 2022,
        },  # Chinchilla
        {
            "title": "Emergent Abilities of Large Language Models",
            "arxiv_id": "2206.07682",
            "year": 2022,
        },
        {
            "title": "PaLM: Scaling Language Modeling with Pathways",
            "arxiv_id": "2204.02311",
            "year": 2022,
        },
    ]

    def __init__(self, s2_client: Optional[SemanticScholarClient] = None):
        """Initialize the collector.

        Args:
            s2_client: Optional Semantic Scholar client for enriching paper data
        """
        self.s2_client = s2_client or SemanticScholarClient()

    def collect_seminal_papers(self) -> List[Dict[str, Any]]:
        """Collect seminal papers list.

        Returns:
            List of paper dictionaries with basic metadata
        """
        papers = []
        for paper_data in self.SEMINAL_PAPERS:
            paper = {
                "title": paper_data["title"],
                "arxiv_id": paper_data.get("arxiv_id"),
                "year": paper_data.get("year"),
                "source": "seminal",
                "authors": None,
                "doi": None,
                "citations": 0,
            }
            papers.append(paper)
        return papers

    def enrich_with_citations(
        self, papers: List[Dict[str, Any]], delay: float = 3.0, max_papers: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Enrich papers with citation data from Semantic Scholar.

        Args:
            papers: List of paper dictionaries
            delay: Delay between API requests (seconds)
            max_papers: Maximum number of papers to enrich (for testing)

        Returns:
            Enriched list of papers
        """
        enriched_papers = []

        papers_to_process = papers[:max_papers] if max_papers else papers

        for i, paper in enumerate(papers_to_process):
            print(f"Enriching paper {i + 1}/{len(papers_to_process)}: {paper['title'][:60]}...")

            # Try to find paper by arXiv ID first, then by title
            s2_paper = None

            if paper.get("arxiv_id"):
                try:
                    s2_paper = self.s2_client.get_paper(
                        f"arXiv:{paper['arxiv_id']}",
                        fields=[
                            "paperId",
                            "title",
                            "abstract",
                            "year",
                            "authors",
                            "citationCount",
                            "influentialCitationCount",
                            "venue",
                            "externalIds",
                        ],
                    )
                except Exception as e:
                    print(f"  ⚠️  Error fetching by arXiv ID: {e}")

            # Fallback to title search if arXiv lookup failed
            if not s2_paper:
                try:
                    search_results = self.s2_client.search(
                        paper["title"],
                        limit=1,
                        fields=[
                            "paperId",
                            "title",
                            "abstract",
                            "year",
                            "authors",
                            "citationCount",
                            "influentialCitationCount",
                            "venue",
                            "externalIds",
                        ],
                    )
                    if search_results.get("data"):
                        s2_paper = search_results["data"][0]
                except Exception as e:
                    print(f"  ⚠️  Error searching by title: {e}")

            # Enrich paper data
            if s2_paper:
                paper["citations"] = s2_paper.get("citationCount", 0)
                paper["s2_paper_id"] = s2_paper.get("paperId")
                paper["abstract"] = s2_paper.get("abstract")
                paper["venue"] = s2_paper.get("venue")

                # Extract authors
                authors = []
                if s2_paper.get("authors"):
                    authors = [a.get("name", "") for a in s2_paper["authors"]]
                    paper["authors"] = ", ".join(authors)

                # Extract external IDs
                external_ids = s2_paper.get("externalIds", {})
                if not paper.get("arxiv_id") and external_ids.get("ArXiv"):
                    paper["arxiv_id"] = external_ids["ArXiv"]
                if not paper.get("doi") and external_ids.get("DOI"):
                    paper["doi"] = external_ids["DOI"]

                # Use S2 year if not set
                if not paper.get("year") and s2_paper.get("year"):
                    paper["year"] = s2_paper["year"]

                print(f"  ✓ Citations: {paper['citations']}, Authors: {len(authors)}")
            else:
                print(f"  ⚠️  Paper not found in Semantic Scholar")

            enriched_papers.append(paper)

            # Rate limiting
            if i < len(papers_to_process) - 1:
                time.sleep(delay)

        return enriched_papers

    def save_to_db(self, papers: List[Dict[str, Any]], db_path: str | Path) -> tuple[int, int]:
        """Save papers to SQLite database.

        Args:
            papers: List of paper dictionaries
            db_path: Path to SQLite database file

        Returns:
            Tuple of (inserted_count, skipped_count)
        """
        with PaperDatabase(db_path) as db:
            inserted, skipped = db.insert_papers_batch(papers)
        return inserted, skipped

    def collect_and_save(
        self,
        db_path: str | Path,
        enrich: bool = True,
        max_papers: Optional[int] = None,
        delay: float = 3.0,
    ) -> Dict[str, Any]:
        """Collect seminal papers and save to database.

        Args:
            db_path: Path to SQLite database
            enrich: Whether to enrich with Semantic Scholar data
            max_papers: Maximum number of papers to process (for testing)
            delay: Delay between API requests (seconds)

        Returns:
            Summary dictionary with statistics
        """
        print("📚 Collecting seminal papers...")
        papers = self.collect_seminal_papers()
        print(f"✓ Collected {len(papers)} seminal papers")

        enriched_count = 0
        if enrich:
            print("\n🔍 Enriching with citation data from Semantic Scholar...")
            papers = self.enrich_with_citations(papers, delay=delay, max_papers=max_papers)
            enriched_count = sum(1 for p in papers if p.get("citations", 0) > 0)
            print(f"✓ Successfully enriched {enriched_count}/{len(papers)} papers")

        print(f"\n💾 Saving to database: {db_path}")
        inserted, skipped = self.save_to_db(papers, db_path)

        summary = {
            "total_papers": len(papers),
            "inserted": inserted,
            "skipped": skipped,
            "enriched": enriched_count if enrich else 0,
            "db_path": str(db_path),
        }

        print(f"\n✅ Complete!")
        print(f"   - Total papers: {summary['total_papers']}")
        print(f"   - Inserted: {summary['inserted']}")
        print(f"   - Skipped (duplicates): {summary['skipped']}")
        if enrich:
            print(f"   - Enriched with citations: {summary['enriched']}")
        print(f"   - Database: {summary['db_path']}")

        return summary
