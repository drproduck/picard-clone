import collections
from typing import Dict, List, Optional, NamedTuple
import transformers.trainer_seq2seq
from transformers.trainer_utils import PredictionOutput, speed_metrics
from datasets.arrow_dataset import Dataset
from datasets.metric import Metric
import numpy as np
import time
from pdb import set_trace

import torch
from torch import nn
from typing import *
import json


class EvalPrediction(NamedTuple):
    predictions: List[str]
    label_ids: np.ndarray
    metas: List[dict]


class Seq2SeqTrainer(transformers.trainer_seq2seq.Seq2SeqTrainer):
    def __init__(
        self,
        metric: Metric,
        *args,
        eval_examples: Optional[Dataset] = None,
        ignore_pad_token_for_loss: bool = True,
        target_with_db_id: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.metric = metric
        self.eval_examples = eval_examples
        self.compute_metrics = self._compute_metrics
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.target_with_db_id = target_with_db_id

    def _compute_metrics(self, eval_prediction: EvalPrediction) -> dict:
        raise NotImplementedError()

    def _post_process_function(
        self, examples: Dataset, features: Dataset, predictions: np.ndarray, stage: str
    ) -> EvalPrediction:
        raise NotImplementedError()

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        eval_examples: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        max_length: Optional[int] = None,
        max_time: Optional[int] = None,
        num_beams: Optional[int] = None,
    ) -> Dict[str, float]:
        self._max_length = max_length
        self._max_time = max_time
        self._num_beams = num_beams

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        if eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples
        start_time = time.time()

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output: PredictionOutput = self.evaluation_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.compute_metrics = compute_metrics

        # We might have removed columns from the dataset so we put them back.
        if isinstance(eval_dataset, Dataset):
            eval_dataset.set_format(
                type=eval_dataset.format["type"],
                columns=list(eval_dataset.features.keys()),
            )
        if eval_examples is not None and eval_dataset is not None and self.compute_metrics is not None:
            eval_preds = self._post_process_function(
                eval_examples,
                eval_dataset,
                output.predictions,
                "eval_{}".format(self.state.epoch),
            )
            output.metrics.update(self.compute_metrics(eval_preds))

        n_samples = len(eval_dataset if eval_dataset is not None else self.eval_dataset)
        output.metrics.update(speed_metrics(metric_key_prefix, start_time, n_samples))

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(output.metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                output.metrics[f"{metric_key_prefix}_{key}"] = output.metrics.pop(key)

        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def predict(
        self,
        test_dataset: Dataset,
        test_examples: Dataset,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        max_length: Optional[int] = None,
        max_time: Optional[int] = None,
        num_beams: Optional[int] = None,
    ) -> PredictionOutput:
        self._max_length = max_length
        self._max_time = max_time
        self._num_beams = num_beams

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        if test_dataset is not None and not isinstance(test_dataset, collections.abc.Sized):
            raise ValueError("test_dataset must implement __len__")

        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output: PredictionOutput = self.evaluation_loop(
                test_dataloader,
                description="Prediction",
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.compute_metrics is not None:
            # We might have removed columns from the dataset so we put them back.
            if isinstance(test_dataset, Dataset):
                test_dataset.set_format(
                    type=test_dataset.format["type"],
                    columns=list(test_dataset.features.keys()),
                )

            eval_preds = self._post_process_function(
                test_examples, test_dataset, output.predictions, metric_key_prefix)
            output.metrics.update(self.compute_metrics(eval_preds))

        output.metrics.update(speed_metrics(metric_key_prefix, start_time, len(test_dataset)))

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(output.metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                output.metrics[f"{metric_key_prefix}_{key}"] = output.metrics.pop(key)

        self.log(output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output

    ####################################################################################################

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            # "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
            "temperature": 1.,
            "top_p": 0.95,
            "do_sample": True,
            "num_return_sequences": 1,
        }

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)

        # prepare generation inputs
        # some encoder-decoder models can have varying encder's and thus
        # varying model input names
        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]

        generated_tokens = self.model.generate(
            generation_inputs,
            **gen_kwargs,
        )

        batch_predictions = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        batch_predictions = np.array(batch_predictions, dtype=object).reshape(-1, gen_kwargs["num_return_sequences"])
        if os.path.exists(f"{self.args.output_dir}/predictions.json"):
            os.remove(f"{self.args.output_dir}/predictions.json")
        with open(f"{self.args.output_dir}/predictions.json", "a") as f:
            for predictions in batch_predictions:
                f.write(json.dumps({
                    "prediction": predictions.tolist(),
                    }) + '\n'
                )

        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            with self.autocast_smart_context_manager():
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
        else:
            labels = None

        return (loss, generated_tokens, labels)

#     def evaluation_loop(
#         self,
#         dataloader: None,
#         description: str,
#         prediction_loss_only: Optional[bool] = None,
#         ignore_keys: Optional[List[str]] = None,
#         metric_key_prefix: str = "eval",
#     ):
#         """
#         Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

#         Works both with or without labels.
#         """
#         args = self.args

#         prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

#         # if eval is called w/o train init deepspeed here
#         if args.deepspeed and not self.deepspeed:

#             # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
#             # from the checkpoint eventually
#             deepspeed_engine, _, _ = deepspeed_init(
#                 self, num_training_steps=0, resume_from_checkpoint=None, inference=True
#             )
#             self.model = deepspeed_engine.module
#             self.model_wrapped = deepspeed_engine
#             self.deepspeed = deepspeed_engine

#         model = self._wrap_model(self.model, training=False)

#         # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
#         # while ``train`` is running, cast it to the right dtype first and then put on device
#         if not self.is_in_train:
#             if args.fp16_full_eval:
#                 model = model.to(dtype=torch.float16, device=args.device)
#             elif args.bf16_full_eval:
#                 model = model.to(dtype=torch.bfloat16, device=args.device)

#         batch_size = dataloader.batch_size

#         # logger.info(f"***** Running {description} *****")
#         # if has_length(dataloader.dataset):
#         #     logger.info(f"  Num examples = {self.num_examples(dataloader)}")
#         # else:
#         #     logger.info("  Num examples: Unknown")
#         # logger.info(f"  Batch size = {batch_size}")

#         model.eval()

#         self.callback_handler.eval_dataloader = dataloader
#         # Do this before wrapping.
#         eval_dataset = dataloader.dataset

#         # if is_torch_tpu_available():
#         #     dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(args.device)

#         if args.past_index >= 0:
#             self._past = None

#         # Initialize containers
#         # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
#         losses_host = None
#         preds_host = None
#         labels_host = None
#         # losses/preds/labels on CPU (final containers)
#         all_losses = None
#         all_preds = None
#         all_labels = None
#         # Will be useful when we have an iterable dataset so don't know its length.

#         observed_num_examples = 0
#         # Main evaluation loop
#         for step, inputs in enumerate(dataloader):
#             # Update the observed num examples
#             observed_batch_size = find_batch_size(inputs)
#             if observed_batch_size is not None:
#                 observed_num_examples += observed_batch_size
#                 # For batch samplers, batch_size is not known by the dataloader in advance.
#                 if batch_size is None:
#                     batch_size = observed_batch_size

#             # Prediction step
#             loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

#             # if is_torch_tpu_available():
#             #     xm.mark_step()

#             # Update containers on host
#             if loss is not None:
#                 losses = self._nested_gather(loss.repeat(batch_size))
#                 losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
#             if labels is not None:
#                 labels = self._pad_across_processes(labels)
#                 labels = self._nested_gather(labels)
#                 labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
#             if logits is not None:
#                 logits = self._pad_across_processes(logits)
#                 logits = self._nested_gather(logits)
#                 if self.preprocess_logits_for_metrics is not None:
#                     logits = self.preprocess_logits_for_metrics(logits, labels)
#                 preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
#             self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

#             # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
#             if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
#                 if losses_host is not None:
#                     losses = nested_numpify(losses_host)
#                     all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
#                 if preds_host is not None:
#                     logits = nested_numpify(preds_host)
#                     all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
#                 if labels_host is not None:
#                     labels = nested_numpify(labels_host)
#                     all_labels = (
#                         labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
#                     )

#                 # Set back to None to begin a new accumulation
#                 losses_host, preds_host, labels_host = None, None, None

#         set_trace()

#         if args.past_index and hasattr(self, "_past"):
#             # Clean the state at the end of the evaluation loop
#             delattr(self, "_past")

#         # Gather all remaining tensors and put them back on the CPU
#         if losses_host is not None:
#             losses = nested_numpify(losses_host)
#             all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
#         if preds_host is not None:
#             logits = nested_numpify(preds_host)
#             all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
#         if labels_host is not None:
#             labels = nested_numpify(labels_host)
#             all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

#         # Number of samples
#         if has_length(eval_dataset):
#             num_samples = len(eval_dataset)
#         # The instance check is weird and does not actually check for the type, but whether the dataset has the right
#         # methods. Therefore we need to make sure it also has the attribute.
#         elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
#             num_samples = eval_dataset.num_examples
#         else:
#             num_samples = observed_num_examples

#         # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
#         # samplers has been rounded to a multiple of batch_size, so we truncate.
#         if all_losses is not None:
#             all_losses = all_losses[:num_samples]
#         if all_preds is not None:
#             all_preds = nested_truncate(all_preds, num_samples)
#         if all_labels is not None:
#             all_labels = nested_truncate(all_labels, num_samples)

#         # Metrics!
#         if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
#             metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
#         else:
#             metrics = {}

#         # To be JSON-serializable, we need to remove numpy types or zero-d tensors
#         metrics = denumpify_detensorize(metrics)

#         if all_losses is not None:
#             metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

#         # Prefix all keys with metric_key_prefix + '_'
#         for key in list(metrics.keys()):
#             if not key.startswith(f"{metric_key_prefix}_"):
#                 metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

#         return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

# def find_batch_size(tensors):
#     """
#     Find the first dimension of a tensor in a nested list/tuple/dict of tensors.
#     """
#     if isinstance(tensors, (list, tuple)):
#         for t in tensors:
#             result = find_batch_size(t)
#             if result is not None:
#                 return result
#     # elif isinstance(tensors, (dict, BatchEncoding)):
#     #     for key, value in tensors.items():
#     #         result = find_batch_size(value)
#     #         if result is not None:
#     #             return result
#     elif isinstance(tensors, torch.Tensor):
#         return tensors.shape[0] if len(tensors.shape) >= 1 else None
#     elif isinstance(tensors, np.ndarray):
#         return tensors.shape[0] if len(tensors.shape) >= 1 else None

# def nested_numpify(tensors):
#     "Numpify `tensors` (even if it's a nested list/tuple of tensors)."
#     if isinstance(tensors, (list, tuple)):
#         return type(tensors)(nested_numpify(t) for t in tensors)
#     t = tensors.cpu()
#     if t.dtype == torch.bfloat16:
#         # As of Numpy 1.21.4, NumPy does not support bfloat16 (see
#         # https://github.com/numpy/numpy/blob/a47ecdea856986cd60eabbd53265c2ca5916ad5d/doc/source/user/basics.types.rst ).
#         # Until Numpy adds bfloat16, we must convert float32.
#         t = t.to(torch.float32)
#     return t.numpy()

# def has_length(dataset):
#     """
#     Checks if the dataset implements __len__() and it doesn't raise an error
#     """
#     try:
#         return len(dataset) is not None
#     except TypeError:
#         # TypeError: len() of unsized object
#         return False