//! Live device-capability probe. Build on any host; run on the target to see
//! what mythd-xpu detects:  `cargo run -p mythd-xpu --example detect`

fn main() {
    match mythd_xpu::DeviceCaps::detect() {
        Ok(c) => {
            println!("DETECTED Intel GPU:");
            println!("  name        = {}", c.name);
            println!("  device_id   = 0x{:04x}", c.device_id);
            println!("  vendor_id   = 0x{:04x}", c.vendor_id);
            println!("  is_xe2      = {}", c.is_xe2);
            println!("  is_b70      = {}", c.is_b70);
            println!("  has_xmx     = {}", c.has_xmx);
            println!("  max_alloc   = {} MB", c.max_alloc_bytes / 1_000_000);
            println!("  precisions  = {:?}", c.native_matrix_precisions);
            println!("  bitnet_int2 = {}", c.supports_bitnet_int2());
        }
        Err(e) => println!("NO DEVICE: {e}"),
    }
}
