import jax
import jax.numpy as jnp
import time
import sys
import os

try:
    import pygame
except ImportError:
    print("PyGame is required for the OS Emulator. Please install it with: pip install pygame")
    sys.exit(1)

# Import the TensorIC Engine
# Assuming this script is run from the tensoric root directory or is in PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from tensoric.staticic import StaticIC
from tensoric.lisp_compiler import parse_to_ast

print("Initializing JAX Pure IC OS Emulator...")

# ==============================================================================
# OS Emulator Config
# ==============================================================================
FPS = 60
FRAME_MS = 1.0 / FPS
HEAP_SIZE = 500_000
STEPS_PER_FRAME = 20_000  # Number of IC interactions the JIT kernel evaluates per frame

# For this demo, we'll map a small display buffer to keep rendering fast.
# In a real IC OS, this would be a massive Tuple-Tree of pixel colors.
DISPLAY_WIDTH = 64
DISPLAY_HEIGHT = 64
DISPLAY_PIXELS = DISPLAY_WIDTH * DISPLAY_HEIGHT
INBOX_SIZE = 128  # Ring buffer for I/O events

# Spatial memory layout offsets (Hardware Memory Map)
DISPLAY_OFFSET = HEAP_SIZE - DISPLAY_PIXELS
INBOX_OFFSET = DISPLAY_OFFSET - INBOX_SIZE


# ==============================================================================
# The OS Topographical Bootstrap (Mock IC Program)
# ==============================================================================
# In a fully realized IC OS, the `.ic` program would structurally route the
# Inbox Tuple Stream to the Display Tuple Tree.
# For this emulator scaffold, we initialize the IC engine and inject synthetic
# state into the designated hardware-mapped memory regions.

ic_engine = StaticIC(heap_size=HEAP_SIZE, enable_gc=True)

# A minimal blank AST just to bootstrap the engine's memory
initial_ast = parse_to_ast("(@ OS_ROOT OS_ROOT)")
ic_engine.load_ast(initial_ast)

# We define a JIT-compiled Frame Evaluate loop.
# This represents the "Hardware Metronome" section.
@jax.jit
def ic_frame_step(heap_state, inbox_events, frame_idx):
    # 1. Hardware Inbox Injection (DMA Emulation)
    # The USB controller (Python) has captured os_events. 
    # We dynamically scatter them into the hardware Inbox memory offset.
    inbox_indices = jnp.arange(INBOX_OFFSET, INBOX_OFFSET + INBOX_SIZE)
    updated_heap_1 = jax.lax.scatter(
        heap_state,
        inbox_indices[:, None],
        inbox_events,
        jax.lax.ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,))
    )
    
    # 2. Continuous IC Topological Evaluation
    def compute_step(carry, _):
        current_heap = carry
        # Real invocation would be: current_heap, _ = ic_engine.rewrite_scan(current_heap, num_steps=1)
        return current_heap, None
    
    # Run the topological evaluation for the frame
    updated_heap_2, _ = jax.lax.scan(compute_step, updated_heap_1, None, length=STEPS_PER_FRAME)
    
    # 3. Emulate the OS Display Tree Logic (Writing to the Framebuffer)
    # Generate dynamic noise based on frame index to prove JAX loop is ticking
    ripple = jax.random.uniform(jax.random.PRNGKey(frame_idx), shape=(DISPLAY_PIXELS,), minval=0, maxval=150, dtype=jnp.uint32)
    display_indices = jnp.arange(DISPLAY_OFFSET, DISPLAY_OFFSET + DISPLAY_PIXELS)
    updated_heap_3 = jax.lax.scatter(
        updated_heap_2,
        display_indices[:, None],
        ripple,
        jax.lax.ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,))
    )

    # Decode Inbox payload offsets and render bright white cursors to visually prove Mouse Input works
    input_offsets = inbox_events & 0x03FFFFFF # Extract 26-bit payload
    valid_clicks = (input_offsets < DISPLAY_PIXELS) & (inbox_events != 0)
    
    updated_heap_final = jax.lax.scatter(
        updated_heap_3,
        (DISPLAY_OFFSET + input_offsets)[:, None],
        jnp.where(valid_clicks, 255, 0), # Max brightness for clicks
        jax.lax.ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,))
    )

    # Return the new OS state and explicitly slice out the Display Buffer (HDMI Sweep)
    display_buffer_out = jnp.take(updated_heap_final, display_indices)
    
    return updated_heap_final, display_buffer_out


# ==============================================================================
# The Python Host Thread (Emulating Legacy USB / HDMI Peripherals)
# ==============================================================================
def run_os():
    pygame.init()
    screen = pygame.display.set_mode((DISPLAY_WIDTH * 8, DISPLAY_HEIGHT * 8)) # Scale up 8x for visibility
    pygame.display.set_caption("Pure IC OS Emulator (JAX)")
    clock = pygame.time.Clock()

    # Initial OS State
    current_heap = ic_engine.heap
    
    # Pre-compile the JIT kernel
    print("JIT Compiling OS Frame Kernel... (This takes a moment)")
    dummy_inbox = jnp.zeros(INBOX_SIZE, dtype=jnp.uint32)
    _ = ic_frame_step(current_heap, dummy_inbox, 0)
    print("OS Kernel Ready. Booting Interactive Loop.")

    inbox_pointer = 0
    inbox_buffer = [0] * INBOX_SIZE
    frame_count = 0

    running = True
    while running:
        # Hardware Timer Metronome implementation
        # The ticking CON(ERA, ERA) is injected every frame
        CON_TAG = 4  # Assuming e.g. 4 is CON
        ERA_TAG = 0  # Assuming 0 is ERA
        tick_node = (CON_TAG << 26) | 0 # Null struct representing un-applied Clock Tuple
        inbox_buffer[inbox_pointer] = tick_node
        inbox_pointer = (inbox_pointer + 1) % INBOX_SIZE

        # 1. Emulate USB Controller (Read Hardware Inputs)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            # Hardware Interrupt: Encode Mouse moves into topological numbers
            if event.type == pygame.MOUSEMOTION or event.type == pygame.MOUSEBUTTONDOWN:
                if pygame.mouse.get_pressed()[0]: # Left click
                    x, y = pygame.mouse.get_pos()
                    # Map scaled coordinates back to IC Display coordinates
                    ic_x, ic_y = x // 8, y // 8
                    
                    if 0 <= ic_x < DISPLAY_WIDTH and 0 <= ic_y < DISPLAY_HEIGHT:
                        mock_node = (5 << 26) | (ic_x + ic_y * DISPLAY_WIDTH) # Tag 5
                        inbox_buffer[inbox_pointer] = mock_node
                        inbox_pointer = (inbox_pointer + 1) % INBOX_SIZE

        # Convert the python list to a JAX array for DMA transfer to the GPU
        jax_inbox_events = jnp.array(inbox_buffer, dtype=jnp.uint32)
        
        # 2. Dispatch the IC OS Frame (The JAX Async Future)
        current_heap, display_future = ic_frame_step(current_heap, jax_inbox_events, frame_count)

        # 3. Emulate HDMI Video Controller (Extract the Display Buffer)
        raw_pixels = jax.device_get(display_future)
        
        # 4. Blast to RGB Monitor (PyGame Surface rendering)
        pixel_array = raw_pixels.reshape((DISPLAY_HEIGHT, DISPLAY_WIDTH))
        surface = pygame.Surface((DISPLAY_WIDTH, DISPLAY_HEIGHT))
        for y in range(DISPLAY_HEIGHT):
            for x in range(DISPLAY_WIDTH):
                val = int(pixel_array[y, x] % 256) 
                surface.set_at((x, y), (val, val, val))
        
        scaled_surface = pygame.transform.scale(surface, (DISPLAY_WIDTH * 8, DISPLAY_HEIGHT * 8))
        screen.blit(scaled_surface, (0, 0))
        pygame.display.flip()

        # Enforce 60FPS Metronome
        clock.tick(FPS)
        frame_count += 1
        
        # Periodically clear the python inbox buffer to stop phantom clicks
        if frame_count % 5 == 0:
             inbox_buffer = [0] * INBOX_SIZE

    pygame.quit()
    print("OS Emulator Shutdown.")

if __name__ == "__main__":
    run_os()
