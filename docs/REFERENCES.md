# References

## Papers

1. **Think Inside the JSON: Reinforcement Strategy for Strict LLM Schema Adherence**
   Bhavik Agarwal, Ishan Joshi, Viktoria Rojkova (2025).
   *arXiv:2502.14905* — [Paper](https://arxiv.org/abs/2502.14905) · [PDF](papers/2502.14905v1.pdf)

   Trains structured reasoning skills of a 1.5B parameter model through a pipeline combining synthetic reasoning dataset construction with custom reward functions under GRPO. First performs R1-style RL on a 20K unstructured-to-structured dataset, then SFT on a 10K reasoning sample dataset for schema adherence. Compares against DeepSeek R1 (671B), distilled variants (Qwen-1.5B/7B), and Gemini 2.0 Flash.

   **Relevance**: Direct inspiration for our reward component architecture — rule-based reward design for JSON schema adherence via RL.

2. **RL-Struct: A Lightweight Reinforcement Learning Framework for Reliable Structured Output in LLMs**
   Ruike Hu, Shulei Wu (2025).
   *arXiv:2512.00319* — [Paper](https://arxiv.org/abs/2512.00319) · [PDF](papers/2512.00319v2.pdf)

   Proposes a lightweight framework using GRPO with a hierarchical reward function to align LLMs with structural constraints. Eliminates the critic network, reducing peak VRAM by 38% vs PPO. Achieves 89.7% structural accuracy and 92.1% validity on complex JSON tasks. Reports an emergent curriculum where the model self-organises learning by prioritising syntax before semantics.

   **Relevance**: Informed our additive reward design, truncation penalty, and validated the curriculum intuition (syntax → semantics).

3. **From Reasoning to Code: GRPO Optimization for Underrepresented Languages**
   Federico Pennino, Bianca Raimondi, Massimo Rondelli, Andrea Gurioli, Maurizio Gabbrielli (2025).
   *arXiv:2506.11027* — [Paper](https://arxiv.org/abs/2506.11027) · [PDF](papers/2506.11027v2.pdf)

   Uses small-scale Qwen 2.5 models with GRPO to enable code generation for underrepresented languages (Prolog) through reasoning-driven feedback integrated into the RL loop. Demonstrates significant improvements in reasoning quality, code accuracy, and logical correctness.

   **Relevance**: Reference for our multi-stage curriculum approach and GRPO applied to structured generation tasks with small models.

4. **ToolRL: Reward is All Tool Learning Needs**
   Cheng Qian, Emre Can Acikgoz, Qi He, Hongru Wang, Xiusi Chen, Dilek Hakkani-Tür, Gokhan Tur, Heng Ji (2025).
   *arXiv:2504.13958* — [Paper](https://arxiv.org/abs/2504.13958) · [PDF](papers/2504.13958v1.pdf)

   First comprehensive study on reward design for tool selection and application within the RL paradigm. Systematically explores reward types, scales, granularity, and temporal dynamics. Trains LLMs with GRPO, achieving 17% improvement over base models and 15% over SFT. Demonstrates that fine-grained, principled reward design is critical for generalisation.

   **Relevance**: Motivated our choice of rule-based rewards over neural reward models and our per-component granular reward design.

5. **Explicit Reasoning Makes Better Judges: A Systematic Study on Accuracy, Efficiency, and Robustness**
   Pratik Jayarao, Himanshu Gupta, Neeraj Varshney, Chaitanya Dwivedi (2025).
   *arXiv:2509.13332* — [Paper](https://arxiv.org/abs/2509.13332) · [PDF](papers/2509.13332v1.pdf)

   Systematic comparison of "thinking" and "non-thinking" LLMs in the LLM-as-a-judge paradigm using Qwen 3 models (0.6B–4B). Thinking models achieve ~10% higher accuracy with under 2x overhead, while augmentation strategies (few-shot, rubrics) deliver modest gains at >8x cost. Bias analyses show thinking models maintain 6% higher consistency under positional, bandwagon, identity, diversity, and random biases.

   **Relevance**: Supports our think vs nothink experimental design — provides evidence that explicit reasoning improves LLM-as-judge accuracy, efficiency, and robustness.

6. **Thinkless: LLM Learns When to Think**
   Gongfan Fang, Xinyin Ma, Xinchao Wang (2025).
   *arXiv:2505.13379* — [Paper](https://arxiv.org/abs/2505.13379) · [PDF](papers/2505.13379v2.pdf)

   Proposes a learnable framework for adaptive reasoning mode selection using control tokens (`<short>` and `<think>`). Introduces Decoupled Group Relative Policy Optimization (DeGRPO), which separates control token loss from response loss to stabilize training and prevent collapse observed in vanilla GRPO. Reduces long-chain thinking usage by 50–90% on benchmarks.

   **Relevance**: Directly relevant to our GRPO training — DeGRPO addresses instabilities observed in vanilla GRPO, and the adaptive think/nothink paradigm parallels our dual-mode experimental setup.

7. **Think-J: Learning to Think for Generative LLM-as-a-Judge**
   Hui Huang, Yancheng He, Hongli Zhou, Rui Zhang, Wei Liu, Weixun Wang, Jiaheng Liu, Wenbo Su (2026).
   *arXiv:2505.14268* — [Paper](https://arxiv.org/abs/2505.14268) · [PDF](papers/2505.14268v2.pdf)

   Improves generative LLM-as-a-Judge by learning judgment thinking traces via RL. Proposes both offline RL (critic-based positive/negative example construction) and online RL (rule-based reward feedback) approaches. Surpasses both generative and classifier-based LLM-Judge without extra human annotations.

   **Relevance**: Validates rule-based reward design for training reasoning capabilities and demonstrates RL-based thinking optimization aligning with our reward-driven GRPO approach.

## Online Resources

8. **Unsloth Documentation**
   [https://unsloth.ai/docs](https://unsloth.ai/docs)
   — Fast LoRA fine-tuning framework; used for accelerated model loading and inference.

9. **AI GRPO — A Deep Dive into Group Relative Policy Optimization**
   Ando AI Blog.
   [https://blog.ando.ai/posts/ai-grpo/](https://blog.ando.ai/posts/ai-grpo/)
   — Conceptual overview of GRPO mechanics and advantage normalisation.

10. **Fine-Tuning GRPO with LLM Judge: From Zero to Production**
    Laurent Bometon, Medium (2025).
    [https://medium.com/@lbometon2/fine-tuning-grpo-with-llm-judge-from-zero-to-production-69a25a4ab3bd](https://medium.com/@lbometon2/fine-tuning-grpo-with-llm-judge-from-zero-to-production-69a25a4ab3bd)
    — Practical GRPO training walkthrough with reward function examples.

11. **Guide to RL Environments for LLMs**
    Patronus AI.
    [https://www.patronus.ai/guide-to-rl-environments](https://www.patronus.ai/guide-to-rl-environments)
    — Survey of RL environments and reward strategies for language model alignment.

## BibTeX

```bibtex
@misc{agarwal2025thinkinsidejsonreinforcement,
      title={Think Inside the JSON: Reinforcement Strategy for Strict LLM Schema Adherence},
      author={Bhavik Agarwal and Ishan Joshi and Viktoria Rojkova},
      year={2025},
      eprint={2502.14905},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.14905},
}

@misc{hu2025rlstructlightweightreinforcementlearning,
      title={RL-Struct: A Lightweight Reinforcement Learning Framework for Reliable Structured Output in LLMs},
      author={Ruike Hu and Shulei Wu},
      year={2025},
      eprint={2512.00319},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2512.00319},
}

@misc{pennino2025reasoningcodegrpooptimization,
      title={From Reasoning to Code: GRPO Optimization for Underrepresented Languages},
      author={Federico Pennino and Bianca Raimondi and Massimo Rondelli and Andrea Gurioli and Maurizio Gabbrielli},
      year={2025},
      eprint={2506.11027},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.11027},
}

@misc{qian2025toolrlrewardtoollearning,
      title={ToolRL: Reward is All Tool Learning Needs},
      author={Cheng Qian and Emre Can Acikgoz and Qi He and Hongru Wang and Xiusi Chen and Dilek Hakkani-Tür and Gokhan Tur and Heng Ji},
      year={2025},
      eprint={2504.13958},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.13958},
}

@misc{jayarao2025explicitreasoningmakesbetter,
      title={Explicit Reasoning Makes Better Judges: A Systematic Study on Accuracy, Efficiency, and Robustness},
      author={Pratik Jayarao and Himanshu Gupta and Neeraj Varshney and Chaitanya Dwivedi},
      year={2025},
      eprint={2509.13332},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2509.13332},
}

@misc{fang2025thinklessllmlearnsthink,
      title={Thinkless: LLM Learns When to Think},
      author={Gongfan Fang and Xinyin Ma and Xinchao Wang},
      year={2025},
      eprint={2505.13379},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.13379},
}

@misc{huang2026thinkjlearningthinkgenerative,
      title={Think-J: Learning to Think for Generative LLM-as-a-Judge},
      author={Hui Huang and Yancheng He and Hongli Zhou and Rui Zhang and Wei Liu and Weixun Wang and Jiaheng Liu and Wenbo Su},
      year={2026},
      eprint={2505.14268},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.14268},
}
```
