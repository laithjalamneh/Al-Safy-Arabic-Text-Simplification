import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_DATASETS_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import traceback
import pandas as pd

print("📂 Loading data first...")
train_df = pd.read_csv("data/train.csv")
val_df   = pd.read_csv("data/val.csv")
print(f"   Train: {train_df.shape[0]} rows")
print(f"   Val  : {val_df.shape[0]} rows")

import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
torch.cuda.empty_cache()

print("imports done ✅")
print(f"💻 Device: {'GPU ✅' if torch.cuda.is_available() else 'CPU ⚠️ (will be slow)'}")


# ── Arabic grammar rules ───────────────────────────────────────────────────
# Simple rule-based checks that don't need camel-tools installed.
# Each rule returns a penalty value between 0.0 and 1.0.
# These run on the decoded output strings during training.

def check_broken_al(text: str) -> float:
    """
    Penalize if (ال) definite article appears with a space before the noun.
    Correct:   الكتاب
    Broken:    ال كتاب
    """
    import re
    broken = re.findall(r'ال\s+\w', text)
    return min(len(broken) * 0.2, 1.0)

def check_no_verb(text: str) -> float:
    """
    Penalize if the output has no Arabic verb.
    Arabic verbs typically start with these root patterns.
    A sentence with no verb is usually a fragment.
    """
    import re
    # Common Arabic verb prefixes (past, present, imperative)
    verb_pattern = re.compile(
        r'\b(يَ|تَ|أَ|نَ|يُ|تُ|أُ|نُ|كَانَ|كَان|هُوَ|هي|كَتَبَ|ذَهَبَ)'
        r'|[يتأن][ا-ي]{2,}'   # present tense pattern
        r'|[ا-ي]{3,}[وا]$',   # past tense plural ending
        re.UNICODE
    )
    if not verb_pattern.search(text):
        return 0.3
    return 0.0

def check_repeated_words(text: str) -> float:
    """
    Penalize if the same word appears 3+ times in a row.
    This catches the garbage repetition loop directly.
    """
    import re
    words = text.split()
    penalty = 0.0
    for i in range(len(words) - 2):
        if words[i] == words[i+1] == words[i+2]:
            penalty += 0.5
    return min(penalty, 1.0)

def arabic_grammar_penalty(text: str) -> float:
    """
    Combines all grammar rules into one scalar penalty.
    Returns a value between 0.0 (perfect) and 1.0 (very broken).
    """
    penalty = 0.0
    penalty += check_broken_al(text)       * 0.3   # weight
    penalty += check_no_verb(text)         * 0.4   # weight
    penalty += check_repeated_words(text)  * 0.3   # weight
    return min(penalty, 1.0)


# ── Custom Trainer ─────────────────────────────────────────────────────────
class SimplificationTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):

        # ── Get model outputs ──────────────────────────────────────────────
        outputs = model(**inputs, output_attentions=True)

        # ── 1. Base loss with label smoothing ─────────────────────────────
        # Label smoothing stops the model being overconfident on each token
        # which reduces repetitive loop generation.
        # Instead of: correct token = 1.0, all others = 0.0
        # We use:     correct token = 0.9, all others = spread across vocab
        SMOOTHING = 0.1
        logits  = outputs.logits                          # (batch, tgt_len, vocab)
        labels  = inputs["labels"]                        # (batch, tgt_len)
        vocab_size = logits.size(-1)

        # Flatten for F.cross_entropy
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs_flat = log_probs.view(-1, vocab_size)
        labels_flat    = labels.view(-1)

        # Standard cross entropy on non-padding tokens
        nll_loss = F.nll_loss(
            log_probs_flat,
            labels_flat,
            ignore_index=-100,
            reduction="mean"
        )

        # Smooth loss — uniform distribution over vocab
        smooth_loss = -log_probs_flat.mean(dim=-1)
        pad_mask    = (labels_flat != -100).float()
        smooth_loss = (smooth_loss * pad_mask).sum() / pad_mask.sum().clamp(min=1)

        base_loss = (1 - SMOOTHING) * nll_loss + SMOOTHING * smooth_loss

        # ── 2. Length penalty ─────────────────────────────────────────────
        # Penalizes output that is SHORTER than input.
        # Lowered weight (0.1) because our new data already has correct
        # length ratios — we don't need to push as hard as before.
        _tokenizer     = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        input_lengths  = (inputs["input_ids"] != _tokenizer.pad_token_id).sum(dim=1).float()
        label_lengths  = (labels != -100).sum(dim=1).float()
        length_ratio   = label_lengths / input_lengths.clamp(min=1)
        length_penalty = torch.relu(1.0 - length_ratio).mean()   # penalize if shorter

        # ── 3. Coverage loss ──────────────────────────────────────────────
        # Forces the decoder to attend to ALL parts of the input,
        # not just the easy/early tokens.
        # Without this the model ignores hard input sections and hallucinates.
        coverage_loss = torch.tensor(0.0, device=logits.device)
        try:
            # cross_attentions: tuple of (batch, heads, tgt_len, src_len) per layer
            # We use the last decoder layer's cross-attention
            if outputs.cross_attentions is not None:
                cross_attn = outputs.cross_attentions[-1]          # last layer
                # Average over heads → (batch, tgt_len, src_len)
                cross_attn = cross_attn.mean(dim=1)
                # Sum over output steps → (batch, src_len)
                # Each source token's total attention received
                coverage   = cross_attn.sum(dim=1)
                # Penalize source tokens that received < 1.0 total attention
                # (meaning the decoder under-attended to them)
                coverage_loss = torch.relu(1.0 - coverage).mean()
        except Exception:
            # If attention extraction fails for any reason, skip silently
            coverage_loss = torch.tensor(0.0, device=logits.device)

        # ── 4. Grammar penalty ────────────────────────────────────────────
        # Decode a sample of outputs and check basic Arabic grammar rules.
        # We only check every 4th step (sampled) to keep training fast —
        # decoding full strings on every step is expensive.
        grammar_loss = torch.tensor(0.0, device=logits.device)
        try:
            # Only run on ~25% of batches to save compute
            if torch.rand(1).item() > 0.75:
                # Greedy decode the logits (no beam search, just argmax)
                pred_ids = logits.argmax(dim=-1)          # (batch, tgt_len)
                penalties = []
                for seq in pred_ids:
                    decoded = _tokenizer.decode(seq, skip_special_tokens=True)
                    if decoded.strip():
                        penalties.append(arabic_grammar_penalty(decoded))
                if penalties:
                    grammar_loss = torch.tensor(
                        sum(penalties) / len(penalties),
                        device=logits.device
                    )
        except Exception:
            grammar_loss = torch.tensor(0.0, device=logits.device)

        # ── 5. Combine all losses ─────────────────────────────────────────
        #
        #   total = base_loss
        #         + 0.10 × length_penalty    (was 0.30 — lowered, data is better now)
        #         + 0.20 × coverage_loss     (new — forces full input attention)
        #         + 0.15 × grammar_loss      (new — penalizes broken Arabic)
        #
        # These weights are the starting point. If val loss is unstable,
        # lower coverage and grammar weights first.
        total_loss = (
            base_loss
            + 0.05 * length_penalty
            + 0.05 * coverage_loss
            + 0.05 * grammar_loss
        )

        if return_outputs:
            return total_loss, outputs
        return total_loss


# ── Training setup ─────────────────────────────────────────────────────────
try:
    MODEL_NAME    = "moussaKam/AraBART"
    OUTPUT_DIR    = "results"
    MAX_INPUT     = 512
    MAX_TARGET    = 256
    BATCH_SIZE    = 1
    EPOCHS        = 60
    LEARNING_RATE = 2e-5

    print(f"🔧 Using model: {MODEL_NAME}")

    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    val_dataset   = Dataset.from_pandas(val_df,   preserve_index=False)

    print("\n⚙️  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"✅ Tokenizer loaded! Pad token ID: {tokenizer.pad_token_id}")

    WHITESPACE_HANDLER = lambda k: " ".join(k.strip().split())

    def preprocess(examples):
        inputs = tokenizer(
            ["تبسيط: " + WHITESPACE_HANDLER(t) for t in examples["text"]],
            max_length=MAX_INPUT,
            truncation=True,
            padding="max_length"
        )
        targets = tokenizer(
            [WHITESPACE_HANDLER(s) for s in examples["summary"]],
            max_length=MAX_TARGET,
            truncation=True,
            padding="max_length"
        )
        inputs["labels"] = targets["input_ids"]
        return inputs

    print("⚙️  Tokenizing dataset (this may take a minute)...")
    train_tokenized = train_dataset.map(
        preprocess,
        batched=True,
        keep_in_memory=True
    )
    val_tokenized = val_dataset.map(
        preprocess,
        batched=True,
        keep_in_memory=True
    )
    print("✅ Tokenization done!")

    print("\n🤖 Loading model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        attn_implementation="eager"   # enables output_attentions=True for coverage loss
    )
    model = model.to("cuda")
    print(f"✅ Model on GPU: {next(model.parameters()).device}")

    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=16,
        learning_rate=LEARNING_RATE,
        dataloader_num_workers=0,
        predict_with_generate=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=3,
        logging_dir="logs",
        logging_steps=50,
        fp16=False,
        bf16=torch.cuda.is_available(),
        optim="adafactor",
        lr_scheduler_type="cosine",
        warmup_steps=100,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none"
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = SimplificationTrainer(
        model=model,
        args=args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=15)]
    )

    print("\n🚀 Starting training...")
    print(f"   Model device : {next(model.parameters()).device}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    print(f"   Loss components: base + length(0.05) + coverage(0.05) + grammar(0.05)")
    print(f"   Scheduler: cosine with 100 warmup steps")
    print(f"   Early stopping: patience=5 epochs")

    trainer.train()

    os.makedirs("model/arabic-simplifier", exist_ok=True)
    model.save_pretrained("model/arabic-simplifier")
    tokenizer.save_pretrained("model/arabic-simplifier")
    print("\n✅ Model saved to model/arabic-simplifier")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    traceback.print_exc()

finally:
    input("\nPress Enter to exit...")