import jax
import jax.numpy as jnp
from os_emulator import ic_frame_step, HEAP_SIZE, INBOX_SIZE
heap = jnp.zeros(HEAP_SIZE, dtype=jnp.uint32)
inbox = jnp.zeros(INBOX_SIZE, dtype=jnp.uint32)
print("Testing JIT compile...")
new_heap, disp = ic_frame_step(heap, inbox, 0)
print("Done!")
