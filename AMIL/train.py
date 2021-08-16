"""

Script adopted from:
    https://github.com/huggingface/transformers/blob/master/examples/run_glue.py

"""

import logging

from tensorboardX import SummaryWriter
from torch.utils.data import RandomSampler
from tqdm import trange
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)
from transformers import BertConfig

from model.model import BertForDistantRE
from utils.train_utils import *

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def train(train_dataset, model):
    tb_writer = SummaryWriter()
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=config.train_batch_size)
    t_total = max(len(train_dataloader) // config.gradient_accumulation_steps * config.num_train_epochs,
                  config.max_steps)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, eps=config.adam_epsilon)
    warmup_steps = int(config.warmup_percent * t_total)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

    # Train
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", config.num_train_epochs)
    logger.info("  Train Batch size = %d", config.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", config.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss, auc, best_f1 = 0.0, 0.0, 0.0, 0.0
    early_stop_counter = config.early_stop
    best_results = dict()
    rel2idx = read_relations(config.relations_file_types)
    na_idx = rel2idx["na"]
    losses, accs, pos_accs = list(), list(), list()
    model.zero_grad()
    train_iterator = trange(0, int(config.num_train_epochs), desc="Epoch", )
    set_seed()

    for _ in train_iterator:  # Epochs
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        avg_loss = AverageMeter()
        avg_acc = AverageMeter()
        avg_pos_acc = AverageMeter()
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(config.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "entity_ids": batch[1],
                "attention_mask": batch[2],
                "labels": batch[4],
                "is_train": True
            }

            loss, logits = model(**inputs)

            # Train results
            preds = torch.argmax(torch.nn.Softmax(-1)(logits), -1)
            acc = float((preds == inputs["labels"]).long().sum()) / inputs["labels"].size(0)
            pos_total = (inputs["labels"] != na_idx).long().sum()
            pos_correct = ((preds == inputs["labels"]).long() * (inputs["labels"] != na_idx).long()).sum()
            if pos_total > 0:
                pos_acc = float(pos_correct) / float(pos_total)
            else:
                pos_acc = 0

            if config.gradient_accumulation_steps > 1:
                loss = loss / config.gradient_accumulation_steps
            # Log
            avg_loss.update(loss.item(), 1)
            avg_acc.update(acc, 1)
            avg_pos_acc.update(pos_acc, 1)

            if global_step % 100 == 0:
                logger.info(" tr_loss = %s", str(avg_loss.avg))
                logger.info(" tr_accuracy = %s", str(avg_acc.avg))
                logger.info(" tr_pos_accuracy = %s", str(avg_pos_acc.avg))

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if config.logging_steps > 0 and global_step % config.logging_steps == 0:
                    logs = {}
                    results = evaluate(model, logger, "dev", ent_types=True)
                    for key, value in results["original"].items():
                        eval_key = "eval_{}".format(key)
                        if key == "R" or key == "P":
                            continue
                        logs[eval_key] = value
                    if results["new_results"]["scikit_f1"] > best_f1:
                        logger.info("  ***  Best ckpt and saved  ***  ")
                        best_results = results
                        best_f1 = results["new_results"]["scikit_f1"]

                        # Save model checkpoint
                        output_dir = os.path.join(config.output_dir, "{}-best-{}".format(global_step, best_f1))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = (model.module if hasattr(model, "module") else model)
                        model_to_save.save_pretrained(output_dir)
                        save_eval_results(results, output_dir, "dev", logger)
                        early_stop_counter = config.early_stop  # reset early stop counter

                    else:
                        early_stop_counter -= 1
                        logger.info("Early stop counter reduced.")

                    loss_scalar = (tr_loss - logging_loss) / config.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    # for key, value in logs.items():
                    #     tb_writer.add_scalar(key, value, global_step)
                    print(json.dumps({**logs, **{"step": global_step}}))

                losses.append(loss.item())
                accs.append(acc)
                pos_accs.append(pos_acc)

                if config.save_steps > 0 and global_step % config.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(config.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (model.module if hasattr(model, "module") else model)
                    model_to_save.save_pretrained(output_dir)

                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

                if early_stop_counter == 0:
                    logger.info("EARLY STOP ON THE TRAINING!")
                    epoch_iterator.close()
                    train_iterator.close()
                    break

            if config.max_steps > 0 and global_step > config.max_steps:
                logger.info("Ending epoch. Reached max num of steps.")
                epoch_iterator.close()
                break

        if config.max_steps > 0 and global_step > config.max_steps:
            logger.info("Ending training. Reached max num of steps.")
            train_iterator.close()
            break

    results = evaluate(model, logger, set_type="dev", prefix="final-{}".format(global_step), ent_types=True)
    if results["new_results"]["scikit_f1"] > best_f1:
        best_results = results
        best_f1 = results["new_results"]["scikit_f1"]

    # Save model checkpoint
    output_dir = os.path.join(config.output_dir, "final-{}".format(global_step))
    model_to_save = (model.module if hasattr(model, "module") else model)
    model_to_save.save_pretrained(output_dir)
    tb_writer.close()
    tr_data = (losses, accs, pos_accs)
    logger.info("***** Best eval F1 : {} *****".format(best_f1))
    logger.info("***** Best dev results *****")
    for key in sorted(best_results["original"].keys()):
        logger.info("  %s = %s", key, str(best_results["original"][key]))

    return global_step, tr_loss / global_step, tr_data


def main():
    os.makedirs(config.output_dir, exist_ok=True)  # Only create model output dir when training
    num_labels = len(read_relations(config.relations_file_types))
    model = BertForDistantRE(BertConfig.from_pretrained(config.pretrained_model_dir), num_labels,
                             bag_attn=config.use_bag_attn)
    model.to(config.device)

    # Training
    train_dataset = load_dataset("train", logger, ent_types=True)
    global_step, tr_loss, tr_data = train(train_dataset, model)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Save final model: with defaults names, you can reload it using from_pretrained()
    logger.info("Saving model checkpoint to %s", config.output_dir)
    model_to_save = (model.module if hasattr(model, "module") else model)
    model_to_save.save_pretrained(config.output_dir)

    # Evaluation
    logger.info("Evaluate the checkpoint: %s", config.test_ckpt)
    model = BertForDistantRE(BertConfig.from_pretrained(config.test_ckpt), num_labels, bag_attn=config.use_bag_attn)
    model.load_state_dict(torch.load(config.test_ckpt + "/pytorch_model.bin"))
    model.to(config.device)
    results = evaluate(model, logger, "test", prefix="TEST", ent_types=True)

    with open(os.path.join(config.test_ckpt, "pr_metrics.txt"), "w") as wf:
        json.dump(str(results), wf)


if __name__ == "__main__":
    main()
