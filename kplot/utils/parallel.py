"""Per-snapshot process pool shared by the analyses."""

# Built-in libraries.
from multiprocessing import Pool, cpu_count
from typing import Callable


def default_workers() -> int:
  """
  Determine the default number of workers.

  Parameters: None
  Returns:
  default number of workers based on cpu_count().
  """
  return min(8, cpu_count())


def pool_map(func: Callable, tasks: list, n_workers: int = None, initializer: Callable = None,
             initargs: tuple = (), progress: bool = False) -> list:
  """
  [func(t) for t in tasks] over `n_workers` processes (default: min(8, cpu_count()));
  `initializer(*initargs)` runs once per worker.

  Parameters:
  func        (Callable): the per-worker analysis function.
  tasks           (list): list of the tasks.
  n_workers        (int): number of parallel workers.
  initializer (Callable): the worker initializer function, if existing.
  initargs       (tuple): initializer function arguments.
  progress        (bool): whether to display a progress-bar.

  Returns:
  list with the results of the workers.
  """
  n_workers = n_workers or default_workers()
  if progress:
    from tqdm import tqdm
  if n_workers > 1:
    with Pool(n_workers, initializer=initializer, initargs=initargs) as pool:
      it = pool.imap_unordered(func, tasks, chunksize=max(1, len(tasks) // (n_workers * 4)))
      return list(tqdm(it, total=len(tasks), desc='Progress') if progress else it)
  if initializer is not None:
    initializer(*initargs)
  return [func(t) for t in (tqdm(tasks, desc='Progress') if progress else tasks)]
